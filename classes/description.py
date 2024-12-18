import math
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Optional

import pandas as pd
import tiktoken
import openai
import numpy as np

import utils.sentences as sentences
from utils.gemini import convert_messages_format

from classes.data_point import Player, Country, Person, PlayerShots
from classes.data_source import PersonStat

import json

from settings import USE_GEMINI

if USE_GEMINI:
    from settings import USE_GEMINI, GEMINI_API_KEY, GEMINI_CHAT_MODEL
else:
    from settings import GPT_BASE, GPT_VERSION, GPT_KEY, GPT_ENGINE

import streamlit as st
import random

openai.api_type = "azure"


class Description(ABC):
    gpt_examples_base = "data/gpt_examples"
    describe_base = "data/describe"

    @property
    @abstractmethod
    def gpt_examples_path(self) -> str:
        """
        Path to excel files containing examples of user and assistant messages for the GPT to learn from.
        """

    @property
    @abstractmethod
    def describe_paths(self) -> Union[str, List[str]]:
        """
        List of paths to excel files containing questions and answers for the GPT to learn from.
        """

    def __init__(self):
        self.synthesized_text = self.synthesize_text()
        self.messages = self.setup_messages()

    def synthesize_text(self) -> str:
        """
        Return a data description that will be used to prompt GPT.

        Returns:
        str
        """

    def get_prompt_messages(self) -> List[Dict[str, str]]:
        """
        Return the prompt that the GPT will see before self.synthesized_text.

        Returns:
        List of dicts with keys "role" and "content".
        """

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a data analysis bot. "
                    "You provide succinct and to the point explanations about data using data. "
                    "You use the information given to you from the data and answers "
                    "to earlier user/assistant pairs to give summaries of players."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about the data for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def get_messages_from_excel(
        self,
        paths: Union[str, List[str]],
    ) -> List[Dict[str, str]]:
        """
        Turn an excel file containing user and assistant columns with str values into a list of dicts.

        Arguments:
        paths: str or list of str
            Path to the excel file containing the user and assistant columns.

        Returns:
        List of dicts with keys "role" and "content".

        """

        # Handle list and str paths arg
        if isinstance(paths, str):
            paths = [paths]
        elif len(paths) == 0:
            return []

        # Concatenate dfs read from paths
        df = pd.read_excel(paths[0])
        for path in paths[1:]:
            df = pd.concat([df, pd.read_excel(path)])

        if df.empty:
            return []

        # Convert to list of dicts
        messages = []
        for i, row in df.iterrows():
            if i == 0:
                messages.append({"role": "user", "content": row["user"]})
            else:
                messages.append({"role": "user", "content": row["user"]})
            messages.append({"role": "assistant", "content": row["assistant"]})

        return messages

    def setup_messages(self) -> List[Dict[str, str]]:
        messages = self.get_intro_messages()
        try:
            paths = self.describe_paths
            messages += self.get_messages_from_excel(paths)
        except (
            FileNotFoundError
        ) as e:  # FIXME: When merging with new_training, add the other exception
            print(e)
        messages += self.get_prompt_messages()

        messages = [
            message for message in messages if isinstance(message["content"], str)
        ]

        try:
            messages += self.get_messages_from_excel(
                paths=self.gpt_examples_path,
            )
        except (
            FileNotFoundError
        ) as e:  # FIXME: When merging with new_training, add the other exception
            print(e)

        messages += [
            {
                "role": "user",
                "content": f"Now do the same thing with the following: ```{self.synthesized_text}```",
            }
        ]
        return messages

    def stream_gpt(self, temperature=1):
        """
        Run the GPT model on the messages and stream the output.

        Arguments:
        temperature: optional float
            The temperature of the GPT model.

        Yields:
            str
        """

        st.expander("Chat transcript", expanded=False).write(self.messages)

        if USE_GEMINI:
            import google.generativeai as genai

            converted_msgs = convert_messages_format(self.messages)

            # # save converted messages to json
            # with open("data/wvs/msgs_0.json", "w") as f:
            #     json.dump(converted_msgs, f)

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_CHAT_MODEL,
                system_instruction=converted_msgs["system_instruction"],
            )
            chat = model.start_chat(history=converted_msgs["history"])
            response = chat.send_message(content=converted_msgs["content"])

            answer = response.text
        else:
            # Use OpenAI API
            openai.api_base = GPT_BASE
            openai.api_version = GPT_VERSION
            openai.api_key = GPT_KEY

            response = openai.ChatCompletion.create(
                engine=GPT_ENGINE,
                messages=self.messages,
                temperature=temperature,
            )

            answer = response["choices"][0]["message"]["content"]

        return answer


class PlayerDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Forward.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/Forward.xlsx"]

    def __init__(self, player: Player):
        self.player = player
        super().__init__()

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a UK-based football scout. "
                    "You provide succinct and to the point explanations about football players using data. "
                    "You use the information given to you from the data and answers "
                    "to earlier user/assistant pairs to give summaries of players."
                ),
            },
            {
                "role": "user",
                "content": "Do you refer to the game you are an expert in as soccer or football?",
            },
            {
                "role": "assistant",
                "content": (
                    "I refer to the game as football. "
                    "When I say football, I don't mean American football, I mean what Americans call soccer. "
                    "But I always talk about football, as people do in the United Kingdom."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about football for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def synthesize_text(self):

        player = self.player
        metrics = self.player.relevant_metrics
        description = f'Here is a statistical description of {player.name}, who played for {player.minutes_played} minutes as a {player.position}. \n\n '

        subject_p, object_p, possessive_p = sentences.pronouns(player.gender)

        for metric in metrics:

            description += f"{subject_p.capitalize()} was "
            description += sentences.describe_level(player.ser_metrics[metric + "_Z"])
            description += " in " + sentences.write_out_metric(metric)
            description += " compared to other players in the same playing position. "

        # st.write(description)

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 4 sentence summary of the player's playing style, strengths and weaknesses. "
            f"The first sentence should use varied language to give an overview of the player. "
            "The second sentence should describe the player's specific strengths based on the metrics. "
            "The third sentence should describe aspects in which the player is average and/or weak based on the statistics. "
            "Finally, summarise exactly how the player compares to others in the same position. "
        )
        return [{"role": "user", "content": prompt}]


class CountryDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/WVS_examples.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/WVS_qualities.xlsx"]

    def __init__(self, country: Country, description_dict, thresholds_dict):
        self.country = country
        self.description_dict = description_dict
        self.thresholds_dict = thresholds_dict

        # read data/wvs/intermediate_data/relevant_questions.json
        with open("data/wvs/intermediate_data/relevant_questions.json", "r") as f:
            self.relevant_questions = json.load(f)

        super().__init__()

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a data analyst and a social scientist. "
                    "You provide succinct and to the point explanations about countries using social factors derived from the World Value Survey. "
                    "You use the information given to you to answer questions about how countries score in various social factors that attempt to measure the social values held by the population of a country."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about the World Value Survey for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def synthesize_text(self):

        description = f"Here is a statistical description of the societal values of {self.country.name.capitalize()}."

        # subject_p, object_p, possessive_p = sentences.pronouns(country.gender)

        for metric in self.country.relevant_metrics:

            description += f"\n\nAccording to the WVS, {self.country.name.capitalize()} was found to "
            description += sentences.describe_level(
                self.country.ser_metrics[metric + "_Z"],
                thresholds=self.thresholds_dict[metric],
                words=self.description_dict[metric],
            )
            description += " compared to other countries in the same wave. "

            if metric in self.country.drill_down_metrics:

                if self.country.ser_metrics[metric + "_Z"] > 0:
                    index = 1
                else:
                    index = 0

                question, value = self.country.drill_down_metrics[metric]
                question, value = question[index], value[index]
                description += "In response to the question '"
                description += self.relevant_questions[metric][question][0]
                description += "', on average participants "
                description += self.relevant_questions[metric][question][1]
                description += " '"
                description += self.relevant_questions[metric][question][2][str(value)]
                description += "' "
                description += self.relevant_questions[metric][question][3]
                description += ". "

        # st.write(description)

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 2 short paragraph summary of the social values held by population of the country. "
            f"The first paragraph should focus on any factors or values for which the country is above or bellow average. If the country is neither above nor below average in any values, mention that. "
            f"The remaining paragraph should mention any specific values or factors that are neither high nor low compared to the average. "
        )
        return [{"role": "user", "content": prompt}]


class PersonDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Forward_bigfive.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/Forward_bigfive.xlsx"]

    def __init__(self, person: Person):
        self.person = person
        super().__init__()

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a recruiter. "
                    "You provide succinct and to the point explanations about a candidate using data.  "
                    "You use the information given to you from the data and answers"
                    "to earlier user/assistant pairs to give summaries of candidates."
                ),
            },
            {
                "role": "user",
                "content": "Do you refer to the candidate as a candidate or a person?",
            },
            {
                "role": "assistant",
                "content": (
                    "I refer to the candidate as a person. "
                    "When I say candidate, I mean person. "
                    "But I always talk about the candidate, as a person."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about a candidate for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def categorie_description(self, value):
        if value <= -2:
            return "The candidate is extremely "
        elif -2 < value <= -1:
            return "The candidate is very "
        elif -1 < value <= -0.5:
            return "The candidate is quite "
        elif -0.5 < value <= 0.5:
            return "The candidate is relatively "
        elif 0.5 < value <= 1:
            return "The candidate is quite "
        elif 1 < value <= 2:
            return "The candidate is very "
        else:
            return "The candidate is extremely "

    def all_max_indices(self, row):
        max_value = row.max()
        return list(row[row == max_value].index)

    def all_min_indices(self, row):
        min_value = row.min()
        return list(row[row == min_value].index)

    def get_description(self, person):
        # here we need the dataset to check the min and max score of the person

        person_metrics = person.ser_metrics
        person_stat = PersonStat()
        questions = person_stat.get_questions()

        name = person.name
        extraversion = person_metrics["extraversion_Z"]
        neuroticism = person_metrics["neuroticism_Z"]
        agreeableness = person_metrics["agreeableness_Z"]
        conscientiousness = person_metrics["conscientiousness_Z"]
        openness = person_metrics["openness_Z"]

        text = []

        # extraversion
        cat_0 = "solitary and reserved. "
        cat_1 = "outgoing and energetic. "

        if extraversion > 0:
            text_t = self.categorie_description(extraversion) + cat_1
            if extraversion > 1:
                index_max = person_metrics[0:10].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2
        else:
            text_t = self.categorie_description(extraversion) + cat_0
            if extraversion < -1:
                index_min = person_metrics[0:10].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        # neuroticism
        cat_0 = "resilient and confident. "
        cat_1 = "sensitive and nervous. "

        if neuroticism > 0:
            text_t = (
                self.categorie_description(neuroticism)
                + cat_1
                + "The candidate tends to feel more negative emotions, anxiety. "
            )
            if neuroticism > 1:
                index_max = person_metrics[10:20].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2

        else:
            text_t = (
                self.categorie_description(neuroticism)
                + cat_0
                + "The candidate tends to feel less negative emotions, anxiety. "
            )
            if neuroticism < -1:
                index_min = person_metrics[10:20].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        # agreeableness
        cat_0 = "critical and rational. "
        cat_1 = "friendly and compassionate. "

        if agreeableness > 0:
            text_t = (
                self.categorie_description(agreeableness)
                + cat_1
                + "The candidate tends to be more cooperative, polite, kind and friendly. "
            )
            if agreeableness > 1:
                index_max = person_metrics[20:30].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2

        else:
            text_t = (
                self.categorie_description(agreeableness)
                + cat_0
                + "The candidate tends to be less cooperative, polite, kind and friendly. "
            )
            if agreeableness < -1:
                index_min = person_metrics[20:30].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        # conscientiousness
        cat_0 = "extravagant and careless. "
        cat_1 = "efficient and organized. "

        if conscientiousness > 0:
            text_t = (
                self.categorie_description(conscientiousness)
                + cat_1
                + "The candidate tends to be more careful or diligent. "
            )
            if conscientiousness > 1:
                index_max = person_metrics[30:40].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2
        else:
            text_t = (
                self.categorie_description(conscientiousness)
                + cat_0
                + "The candidate tends to be less careful or diligent. "
            )
            if conscientiousness < -1:
                index_min = person_metrics[30:40].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        # openness
        cat_0 = "consistent and cautious. "
        cat_1 = "inventive and curious. "

        if openness > 0:
            text_t = (
                self.categorie_description(openness)
                + cat_1
                + "The candidate tends to be more open. "
            )
            if openness > 1:
                index_max = person_metrics[40:50].idxmax()
                text_2 = (
                    "In particular they said that " + questions[index_max][0] + ". "
                )
                text_t += text_2
        else:
            text_t = (
                self.categorie_description(openness)
                + cat_0
                + "The candidate tends to be less open. "
            )
            if openness < -1:
                index_min = person_metrics[40:50].idxmin()
                text_2 = (
                    "In particular they said that " + questions[index_min][0] + ". "
                )
                text_t += text_2
        text.append(text_t)

        text = "".join(text)
        text = text.replace(",", "")
        return text

    def synthesize_text(self):
        person = self.person
        metrics = self.person.ser_metrics
        description = self.get_description(person)

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 4 sentence summary of the person's personality, strengths and weaknesses. "
            f"The first sentence should use varied language to give an overview of the person. "
            "The second sentence should describe the person's specific strengths based on the metrics. "
            "The third sentence should describe aspects in which the person is average and/or weak based on the statistics. "
            "Finally, summarise exactly how the person compares to others in the same position. "
        )
        return [{"role": "user", "content": prompt}]

class ShotsDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Shots.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/Shots.xlsx"]

    def __init__(self, player: PlayerShots):
        self.player = player
        super().__init__()

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a UK-based football scout. "
                    "You provide succinct and to the point explanations about football players using data. "
                    "You use the information given to you from the data and answers "
                    "to earlier user/assistant pairs to give summaries of players shots during the Copa America 2024."
                ),
            },
            {
                "role": "user",
                "content": "Do you refer to the game you are an expert in as soccer or football?",
            },
            {
                "role": "assistant",
                "content": (
                    "I refer to the game as football. "
                    "When I say football, I don't mean American football, I mean what Americans call soccer. "
                    "But I always talk about football, as people do in the United Kingdom."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about football for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def synthesize_text(self):

        player = self.player
        df = player.ser_metrics
        description = f'Here is a description of {player.name}\'s shots during Copa America 2024. \n\n '
        description += f'He attempted '
        description += f'{df["total_shots"]} shots during the tournament. '
        #Pitch zone
        if df["outside_shots"] > 0 :
            to_be = 'were' if df["outside_shots"] > 1 else 'was'
            description += f'{df["outside_shots"]} of his shots {to_be} from outside the box. '
        if df["inside_shots"] > 0 :
            to_be = 'were' if df["inside_shots"] > 1 else 'was'
            description += f'{df["inside_shots"]} of his shots {to_be} from inside the box. '

        if df['avg_distance'] >= 0:
            description += f'On average, his shots were {df["avg_distance"]:.2f} meters away from the goal. '

        #Part of the body
        if df["right_shots"] > 0 :
            to_be = 'were' if df["right_shots"] > 1 else 'was'
            description += f'{df["right_shots"]} of his shots {to_be} made with his right foot. '
        if df["left_shots"] > 0 :
            to_be = 'were' if df["left_shots"] > 1 else 'was'
            description += f'{df["left_shots"]} of his shots {to_be} made with his left foot. '
        if df["head_shots"] > 0 :
            to_be = 'were' if df["head_shots"] > 1 else 'was'
            description += f'{df["head_shots"]} of his shots {to_be} made with his head. '
        if df["other_part_shots"] > 0 :
            to_be = 'were' if df["other_part_shots"] > 1 else 'was'
            description += f'{df["other_part_shots"]} of his shots {to_be} made with other part of the body. '

        #Type of shot
        if df["penalty_shots"] > 0 :
            to_be = 'were' if df["penalty_shots"] > 1 else 'was'
            type_name = "penalties" if df["penalty_shots"] > 1 else "a penalty"
            description += f'{df["penalty_shots"]} of his shots {to_be} {type_name}. '
        if df["free_kick_shots"] > 0 :
            to_be = 'were' if df["free_kick_shots"] > 1 else 'was'
            type_name = "free kicks" if df["free_kick_shots"] > 1 else "a free kick"
            description += f'{df["free_kick_shots"]} of his shots {to_be} {type_name}. ' 
        if df["open_play_shots"] > 0 :
            to_be = 'were' if df["open_play_shots"] > 1 else 'was'
            description += f'{df["open_play_shots"]} of his shots {to_be} from open play. '

        #Outcome
        if df["goals"] > 0 :
            type_name = "goals" if df["goals"] > 1 else "goal"
            description += f'He socred {df["goals"]} {type_name}. '
        if df["saved_shots"] > 0 :
            to_be = 'were' if df["saved_shots"] > 1 else 'was'
            description += f'{df["saved_shots"]} of his shots {to_be} saved by the goalkeeper. '
        if df["blocked_shots"] > 0 :
            to_be = 'were' if df["blocked_shots"] > 1 else 'was'
            description += f'{df["blocked_shots"]} of his shots {to_be} blocked by a defender. '
        if df["off_t_shots"] > 0 :
            to_be = 'were' if df["off_t_shots"] > 1 else 'was'
            description += f'{df["off_t_shots"]} of his shots {to_be} off target or hit the posts. '    

        #Inside the box - Outcome
        description += f'From his shots inside the box, '
        if df["inside_goals"] > 0 :
            type_name = "goals" if df["inside_goals"] > 1 else "goal"
            description += f'he socred {df["inside_goals"]} {type_name}, '
        if df["inside_saved_shots"] > 0 :
            to_be = 'were' if df["inside_saved_shots"] > 1 else 'was'
            description += f'{df["inside_saved_shots"]} {to_be} saved by the goalkeeper, '
        if df["inside_blocked_shots"] > 0 :
            to_be = 'were' if df["inside_blocked_shots"] > 1 else 'was'
            description += f'{df["inside_blocked_shots"]} {to_be} blocked by a defender, '
        if df["inside_off_t_shots"] > 0 :
            to_be = 'were' if df["inside_off_t_shots"] > 1 else 'was'
            description += f'{df["inside_off_t_shots"]} {to_be} off target or hit the posts. '
        
        if description.endswith(", "):
            description = description[:-2] + ". "

        #Outside the box - Outcome
        description += f'From his shots outside the box, '
        if df["outside_goals"] > 0 :
            type_name = "goals" if df["outside_goals"] > 1 else "goal"
            description += f'he socred {df["outside_goals"]} {type_name}, '
        if df["outside_saved_shots"] > 0 :
            to_be = 'were' if df["outside_saved_shots"] > 1 else 'was'
            description += f'{df["outside_saved_shots"]} {to_be} saved by the goalkeeper, '
        if df["outside_blocked_shots"] > 0 :
            to_be = 'were' if df["outside_blocked_shots"] > 1 else 'was'
            description += f'{df["outside_blocked_shots"]} {to_be} blocked by a defender, '
        if df["outside_off_t_shots"] > 0 :
            to_be = 'were' if df["outside_off_t_shots"] > 1 else 'was'
            description += f'{df["outside_off_t_shots"]} {to_be} off target or hit the posts. '
        
        if description.endswith(", "):
            description = description[:-2] + ". "

        #Free kicks - Outcome
        description += f'From his free kick shtots, '
        if df["free_kick_goals"] > 0 :
            type_name = "goals" if df["free_kick_goals"] > 1 else "goal"
            description += f'he socred {df["free_kick_goals"]} {type_name}, '
        if df["free_kick_saved_shots"] > 0 :
            to_be = 'were' if df["free_kick_saved_shots"] > 1 else 'was'
            description += f'{df["free_kick_saved_shots"]} {to_be} saved by the goalkeeper, '
        if df["free_kick_blocked_shots"] > 0 :
            to_be = 'were' if df["free_kick_blocked_shots"] > 1 else 'was'
            description += f'{df["free_kick_blocked_shots"]} {to_be} blocked by a defender, '
        if df["free_kick_off_t_shots"] > 0 :
            to_be = 'were' if df["free_kick_off_t_shots"] > 1 else 'was'
            description += f'{df["free_kick_off_t_shots"]} {to_be} off target or hit the posts. '
        
        if description.endswith(", "):
            description = description[:-2] + ". "

        description += f'From his penalty shots, '
        #Penalty - Outcome
        if df["penalty_goals"] > 0 :
            type_name = "goals" if df["penalty_goals"] > 1 else "goal"
            description += f'he socred {df["penalty_goals"]} {type_name}, '
        if df["penalty_saved_shots"] > 0 :
            to_be = 'were' if df["penalty_saved_shots"] > 1 else 'was'
            description += f'{df["penalty_saved_shots"]} {to_be} saved by the goalkeeper, '
        if df["penalty_blocked_shots"] > 0 :
            to_be = 'were' if df["penalty_blocked_shots"] > 1 else 'was'
            description += f'{df["penalty_blocked_shots"]} {to_be} blocked by a defender, '
        if df["penalty_off_t_shots"] > 0 :
            to_be = 'were' if df["penalty_off_t_shots"] > 1 else 'was'
            description += f'{df["penalty_off_t_shots"]} {to_be} off target or hit the posts. '
        
        if description.endswith(", "):
            description = description[:-2] + ". "

        #Goals -Pitch zone
        if df["outside_goals"] > 0 :
            to_be = 'were' if df["outside_goals"] > 1 else 'was'
            description += f'{df["outside_goals"]} of his goals {to_be} scored from outside the box. '
        if df["inside_goals"] > 0 :
            to_be = 'were' if df["inside_goals"] > 1 else 'was'
            description += f'{df["inside_goals"]} of his goals {to_be} scored from inside the box. '

        #Part of the body
        if df["right_goals"] > 0 :
            to_be = 'were' if df["right_goals"] > 1 else 'was'
            description += f'{df["right_goals"]} of his goals {to_be} scored using his right foot. '
        if df["left_goals"] > 0 :
            to_be = 'were' if df["left_goals"] > 1 else 'was'
            description += f'{df["left_goals"]} of his goals {to_be} scored using his left foot. '
        if df["head_goals"] > 0 :
            to_be = 'were' if df["head_goals"] > 1 else 'was'
            description += f'{df["head_goals"]} of his goals {to_be} scored using his head. '
        if df["other_part_goals"] > 0 :
            to_be = 'were' if df["other_part_goals"] > 1 else 'was'
            description += f'{df["other_part_goals"]} of his goals {to_be} scored using other part of the body. '

        #Type of shot
        if df["penalty_goals"] > 0 :
            to_be = 'were' if df["penalty_goals"] > 1 else 'was'
            type_name = "penalties" if df["penalty_goals"] > 1 else "a penalty"
            description += f'{df["penalty_goals"]} of his goals {to_be} scored from {type_name}. '
        if df["free_kick_goals"] > 0 :
            to_be = 'were' if df["free_kick_goals"] > 1 else 'was'
            type_name = "free kicks" if df["free_kick_goals"] > 1 else "a free kick"
            description += f'{df["free_kick_goals"]} of his goals {to_be} scored from {type_name}. '
        if df["open_play_goals"] > 0 :
            to_be = 'were' if df["open_play_goals"] > 1 else 'was'
            description += f'{df["open_play_goals"]} of his goals {to_be} scored from open play. '

        if df["goals"] >= 0 and df["total_xG"] >= 0:
            performance_description =  "he socred less goals than expected, so he underperformed" if df["goals"] < df["total_xG"] else "he scored more goals than expected, so he overperformed"
            scored_description = 'goals' if df["goals"] > 1 else 'goal'
            description += f'His accumulated expected goal (excluding penalties) was {df["total_xG"]:.2f}, '
            description += f'since he scored {df["goals"]} {scored_description}, {performance_description}. '
        
            

        if df['xG_per_shot'] >= 0 and df["max_xg"] >= 0:
            result_description = "off target or hit the posts" if df["max_xg_outcome"] == "Off T" or df["max_xg_outcome"] == "Wayward" or df["max_xg_outcome"] == "Post" else "saved" if df["max_xg_outcome"] == "Saved" or df["max_xg_outcome"] == "Saved to Post" else df["max_xg_outcome"]
            description += f'His xG per shot was ({df["xG_per_shot"]:.2f}), and '
            description += f'his shot with the highest expected goal ({df["max_xg"]:.2f}) was {result_description}.'

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the player shots description enclosed with ``` to give a concise, 5 sentence summary of the player's shots during the tournament. "
            f"The first sentence should use varied language to give an overview of the player performance. "
            "The second sentence should describe the player's specific strengths based on the shots. "
            "The third sentence should describe where the player performed well based on the shots he made. "
            "The fourth sentence should describe where the player performed wrong based on the shots he made. "
            "Finally, summarize exactly how the player performed based on the shots."
        )
        return [{"role": "user", "content": prompt}]

