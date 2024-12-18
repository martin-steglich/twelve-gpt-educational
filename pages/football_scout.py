"""
Entrypoint for streamlit app.
Runs top to bottom every time the user interacts with the app (other than imports and cached functions).
"""

# Library imports
import traceback
import copy

import streamlit as st

from classes.data_source import PlayerShotsStats
from classes.data_point import PlayerShots
from classes.visual import ShotsPlot
from classes.description import (
    ShotsDescription,
)
from classes.chat import PlayerShotsChat

from utils.page_components import (
    add_common_page_elements,
    #     select_player,
    #     create_chat,
)

from utils.utils import (
    select_player_shots,
    create_chat,
)


# def show():
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

players_shots = PlayerShotsStats()

# Define the metrics we are interested in and calculates them
metrics = [
    "total_shots"
    "outside_shots"
    "inside_shots"
    "right_shots"
    "left_shots"
    "head_shots"
    "other_part_shots"
    "penalty_shots"
    "free_kick_shots"
    "open_play_shots"
    "goals"
    "saved_shots"
    "blocked_shots"
    "off_t_shots"
    "inside_goals"
    "inside_saved_shots"
    "inside_blocked_shots"
    "inside_off_t_shots"
    "outside_goals"
    "outside_saved_shots"
    "outside_blocked_shots"
    "outside_off_t_shots"
    "free_kick_goals"
    "free_kick_saved_shots"
    "free_kick_blocked_shots"
    "free_kick_off_t_shots"
    "penalty_goals"
    "penalty_saved_shots"
    "penalty_blocked_shots"
    "penalty_off_t_shots"
    "outside_goals"
    "inside_goals"
    "right_goals"
    "left_goals"
    "head_goals"
    "other_part_goals"
    "penalty_goals"
    "free_kick_goals"
    "open_play_goals"
    "accumulated_npxg"
    "max_npxg"
    "result"
]


# Now select the focal player
player = select_player_shots(sidebar_container, players_shots)

st.write(
    "This app can only handle three or four users at a time. Please [download](https://github.com/soccermatics/twelve-gpt-educational) and run on your own computer with your own Gemini key."
)

# Read in model card text
with open("model cards/model-card-football-scout.md", "r", encoding="utf8") as file:
    # Read the contents of the file
    model_card_text = file.read()


st.expander("Model card for Football Scout", expanded=False).markdown(model_card_text)

st.expander("Dataframe used", expanded=False).write(players_shots.df)

# Chat state hash determines whether or not we should load a new chat or continue an old one
# We can add or remove variables to this hash to change conditions for loading a new chat
to_hash = (player.id,)
# Now create the chat as type PlayerChat
chat = create_chat(to_hash, PlayerShotsChat, player, players_shots)

# Now we want to add basic content to chat if it's empty
if chat.state == "empty":

    # Make a plot of the distribution of the metrics for all players
    # We reverse the order of the elements in metrics for plotting (because they plot from bottom to top)
    visual = ShotsPlot()
    visual.add_title_from_player(player)
    # visual.add_players(players, metrics=metrics)
    visual.add_player_shots(player)
    visual.add_stats(player)
    visual.add_footer()
    

    # Now call the description class to get the summary of the player
    description = ShotsDescription(player)
    summary = description.stream_gpt()

    # Add the visual and summary to the chat
    chat.add_message(
        "Please can you analyse " + player.name + " shots for me?",
        role="user",
        user_only=False,
        visible=False,
    )
    visual.show_plot()
    # chat.add_message(visual)
    chat.add_message(summary)

    chat.state = "default"

# Now we want to get the user input, display the messages and save the state
chat.get_input()
chat.display_messages()
chat.save_state()
