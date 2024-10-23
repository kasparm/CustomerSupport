from datetime import datetime
import os
import uuid
import time

import chainlit as cl
import logging
logging.basicConfig(
    filename="app.log",
    encoding="utf-8",
    filemode="a",
    level=logging.DEBUG,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)

from dotenv import load_dotenv
load_dotenv(override=True)

from typing import Annotated
from typing_extensions import TypedDict

from db_setup import db
from car_rental import (
    update_car_rental, book_car_rental, cancel_car_rental, search_car_rentals
)
from excursion_reservation import (
    cancel_excursion, update_excursion, book_excursion, search_trip_recommendations
)
from flight_reservation import (
    search_flights, update_ticket_to_new_flight, fetch_user_flight_information, cancel_ticket
)
from hotel_reservation import (
    cancel_hotel, update_hotel, book_hotel, search_hotels
)

from langgraph.graph.message import AnyMessage, add_messages
from utilities import create_tool_node_with_fallback, _print_event, handle_tool_error

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

_printed = set()
_config = None
_thread_id = None

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

@cl.on_chat_start
async def on_chat_start():
    logging.info("Chat setup")
    # Haiku is faster and cheaper, but less accurate
    # llm = ChatAnthropic(model="claude-3-haiku-20240307")
    # llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
    API_HOST = os.getenv("API_HOST")
    if API_HOST == "github":
        llm = ChatOpenAI(
        openai_api_base="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
        model=os.getenv("GITHUB_MODEL")
    )
    else:
        raise ValueError("Unsupported API host")

    # You could swap LLMs, though you will likely want to update the prompts when
    # doing so!
    # from langchain_openai import ChatOpenAI

    # llm = ChatOpenAI(model="gpt-4-turbo-preview")

    primary_assistant_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful customer support assistant for Swiss Airlines. "
                "Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
                "When searching, be persistent. Expand your query bounds if the first search returns no results. "
                "If a search comes up empty, expand your search before giving up."
                "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
                "\nCurrent time: {time}.",
            ),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=datetime.now())

    # lookup_policy,
    part_1_tools = [
        TavilySearchResults(max_results=1),
        fetch_user_flight_information,
        search_flights,
        update_ticket_to_new_flight,
        cancel_ticket,
        search_car_rentals,
        book_car_rental,
        update_car_rental,
        cancel_car_rental,
        search_hotels,
        book_hotel,
        update_hotel,
        cancel_hotel,
        search_trip_recommendations,
        book_excursion,
        update_excursion,
        cancel_excursion,
    ]
    part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)

    builder = StateGraph(State)

    # Define nodes: these do the work
    builder.add_node("assistant", Assistant(part_1_assistant_runnable))
    builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # The checkpointer lets the graph persist its state
    # this is a complete memory for the entire graph.
    memory = MemorySaver()
    part_1_graph = builder.compile(checkpointer=memory)


    # Update with the backup file so we can restart from the original place in each section
    logging.info("Start db update")
    db.update_dates()
    logging.info("DB update done")
    
    _thread_id = cl.context.session.thread_id
    # save graph and state to the user session
    cl.user_session.set("graph", part_1_graph)
    logging.info("Setup Done")


@cl.on_message
async def on_chat_message(message: cl.Message):
    logging.info("Chat message start")
    part_1_graph: Runnable = cl.user_session.get("graph")

    ui_message = cl.Message(content="")
    await ui_message.send()
    logging.info("Message received")


    _config = {
        "configurable": {
            # The passenger_id is used in our flight tools to
            # fetch the user's flight information
            "passenger_id": "3442 587242",
            # Checkpoints are accessed by thread_id
            "thread_id": _thread_id,
        }
    }

    events = part_1_graph.stream(
        {"messages": ("user", message.content)}, _config, stream_mode="values"
    )
    for event in events:
        #_print_event(event, _printed)
        # print(f"event: {event}")
        #print(type(event))
        current_state = event.get("dialog_state")
        #print(f"current_state: {current_state}")
        message = event.get("messages")
        print(f"message: {message}")
        print(f"message -1 content: {message[-1].content}")
        print(f"message type: {message[-1].type}")
        if message[-1].type == "ai":
            await ui_message.stream_token(token=message[-1].content)
    await ui_message.update()
    logging.info("Message response delivered")

    # Sleep for a bit to allow the API to work with rate limits
   # time.sleep(10)


def run_with_preset_conversation():
     # Let's create an example conversation a user might have with the assistant
    tutorial_questions = [
        "Hi there, what time is my flight?",
        "Am i allowed to update my flight to something sooner? I want to leave later today.",
        "Update my flight to sometime next week then",
        "The next available option is great",
        "what about lodging and transportation?",
        "Yeah i think i'd like an affordable hotel for my week-long stay (7 days). And I'll want to rent a car.",
        "OK could you place a reservation for your recommended hotel? It sounds nice.",
        "yes go ahead and book anything that's moderate expense and has availability.",
        "Now for a car, what are my options?",
        "Awesome let's just get the cheapest option. Go ahead and book for 7 days",
        "Cool so now what recommendations do you have on excursions?",
        "Are they available while I'm there?",
        "interesting - i like the museums, what options are there?",
        "OK great pick one and book it for my second day there.",
    ]

    # Update with the backup file so we can restart from the original place in each section
    db.update_dates()
    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            # The passenger_id is used in our flight tools to
            # fetch the user's flight information
            "passenger_id": "3442 587242",
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }

    _printed = set()
    for question in tutorial_questions:
        events = part_1_graph.stream(
            {"messages": ("user", question)}, config, stream_mode="values"
        )
        for event in events:
            _print_event(event, _printed)

        # Sleep for a bit to allow the API to work with rate limits
        time.sleep(10)