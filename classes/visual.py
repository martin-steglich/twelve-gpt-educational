import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


from utils.sentences import format_metric
from classes.data_point import Player, Country, Person, PlayerShots
from classes.data_source import PlayerStats, CountryStats, PersonStat
from typing import Union
from matplotlib.colors import ListedColormap, to_rgba
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import matplotlib.cm as cm
import matplotlib.font_manager as font_manager
from PIL import Image


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def rgb_to_color(rgb_color: tuple, opacity=1):
    return f"rgba{(*rgb_color, opacity)}"


def tick_text_color(color, text, alpha=1.0):
    # color: hexadecimal
    # alpha: transparency value between 0 and 1 (default is 1.0, fully opaque)
    s = (
        "<span style='color:rgba("
        + str(int(color[1:3], 16))
        + ","
        + str(int(color[3:5], 16))
        + ","
        + str(int(color[5:], 16))
        + ","
        + str(alpha)
        + ")'>"
        + str(text)
        + "</span>"
    )
    return s

def create_pastel_cmap(base_cmap, n_colors=10, blend_ratio=0.5):
    """
    Modify a base colormap to produce pastel colors.

    Parameters:
        base_cmap (str): The name of the base colormap.
        n_colors (int): Number of discrete colors.
        blend_ratio (float): Blend ratio with white (0 = base color, 1 = white).

    Returns:
        ListedColormap: A colormap with pastel colors.
    """
    base = cm.get_cmap(base_cmap, n_colors)  # Get the base colormap
    pastel_colors = []

    for i in range(n_colors):
        color = np.array(base(i)[:3])  # Get RGB components
        pastel_color = color * (1 - blend_ratio) + np.array([1, 1, 1]) * blend_ratio  # Blend with white
        pastel_colors.append((*pastel_color, 1))  # Add alpha = 1

    return ListedColormap(pastel_colors)

def get_marker(sub_type_name, body_part_name):
  if sub_type_name == 'Penalty':
    return "o"
  
  if sub_type_name == 'Free Kick':
    return "h"
  
  if body_part_name == 'Head':
    return "D"
  
  return "s"

class Visual:
    # Can't use streamlit options due to report generation
    dark_green = hex_to_rgb(
        "#002c1c"
    )  # hex_to_rgb(st.get_option("theme.secondaryBackgroundColor"))
    medium_green = hex_to_rgb("#003821")
    bright_green = hex_to_rgb(
        "#00A938"
    )  # hex_to_rgb(st.get_option("theme.primaryColor"))
    bright_orange = hex_to_rgb("#ff4b00")
    bright_yellow = hex_to_rgb("#ffcc00")
    bright_blue = hex_to_rgb("#0095FF")
    white = hex_to_rgb("#ffffff")  # hex_to_rgb(st.get_option("theme.backgroundColor"))
    gray = hex_to_rgb("#808080")
    black = hex_to_rgb("#000000")
    light_gray = hex_to_rgb("#d3d3d3")
    table_green = hex_to_rgb("#009940")
    table_red = hex_to_rgb("#FF4B00")
    

    def __init__(self, pdf=False, plot_type="scout"):
        self.pdf = pdf
        if pdf:
            self.font_size_multiplier = 1.4
        else:
            self.font_size_multiplier = 1.0
        # self.fig = go.Figure()
        
        # self._setup_styles()
        # self.plot_type = plot_type

        # if plot_type == "scout":
        #     self.annotation_text = (
        #         "<span style=''>{metric_name}: {data:.2f} per 90</span>"
        #     )
        # else:
        #     # self.annotation_text = "<span style=''>{metric_name}: {data:.0f}/66</span>"  # TODO: this text will not automatically update!
        #     self.annotation_text = "<span style=''>{metric_name}: {data:.2f}</span>"

    def show(self):
        st.plotly_chart(
            self.fig,
            config={"displayModeBar": False},
            height=500,
            use_container_width=True,
        )

    def _setup_styles(self):
        side_margin = 60
        top_margin = 75
        pad = 16
        self.fig.update_layout(
            autosize=True,
            height=500,
            margin=dict(l=side_margin, r=side_margin, b=70, t=top_margin, pad=pad),
            paper_bgcolor=rgb_to_color(self.dark_green),
            plot_bgcolor=rgb_to_color(self.dark_green),
            legend=dict(
                orientation="h",
                font={
                    "color": rgb_to_color(self.white),
                    "family": "Gilroy-Light",
                    "size": 11 * self.font_size_multiplier,
                },
                itemclick=False,
                itemdoubleclick=False,
                x=0.5,
                xanchor="center",
                y=-0.2,
                yanchor="bottom",
                valign="middle",  # Align the text to the middle of the legend
            ),
            xaxis=dict(
                tickfont={
                    "color": rgb_to_color(self.white, 0.5),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            ),
        )

    def add_title(self, title, subtitle):
        self.title = title
        self.subtitle = subtitle
        self.fig.update_layout(
            title={
                "text": f"<span style='font-size: {15*self.font_size_multiplier}px'>{title}</span><br>{subtitle}",
                "font": {
                    "family": "Gilroy-Medium",
                    "color": rgb_to_color(self.white),
                    "size": 12 * self.font_size_multiplier,
                },
                "x": 0.05,
                "xanchor": "left",
                "y": 0.93,
                "yanchor": "top",
            },
        )

    def add_low_center_annotation(self, text):
        self.fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.07,
            text=text,
            showarrow=False,
            font={
                "color": rgb_to_color(self.white, 0.5),
                "family": "Gilroy-Light",
                "size": 12 * self.font_size_multiplier,
            },
        )

    def show(self):
        st.plotly_chart(
            self.fig,
            config={"displayModeBar": False},
            height=500,
            use_container_width=True,
        )

    def close(self):
        pass


class DistributionPlot(Visual):
    def __init__(self, columns, labels=None, *args, **kwargs):
        self.empty = True
        self.columns = columns
        self.marker_color = (
            c for c in [Visual.white, Visual.bright_yellow, Visual.bright_blue]
        )
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        super().__init__(*args, **kwargs)
        if labels is not None:
            self._setup_axes(labels)
        else:
            self._setup_axes()

    def _setup_axes(self, labels=["Worse", "Average", "Better"]):
        self.fig.update_xaxes(
            range=[-4, 4],
            fixedrange=True,
            tickmode="array",
            tickvals=[-3, 0, 3],
            ticktext=labels,
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
        )

    def add_group_data(self, df_plot, plots, names, legend, hover="", hover_string=""):
        showlegend = True

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            temp_df = pd.DataFrame(df_plot[col + hover])
            temp_df["name"] = metric_name

            self.fig.add_trace(
                go.Scatter(
                    x=df_plot[col + plots],
                    y=np.ones(len(df_plot)) * i,
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.bright_green, opacity=0.2),
                        "size": 10,
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=names,
                    customdata=df_plot[col + hover],
                    name=legend,
                    showlegend=showlegend,
                )
            )
            showlegend = False

    def add_data_point(
        self, ser_plot, plots, name, hover="", hover_string="", text=None
    ):
        if text is None:
            text = [name]
        elif isinstance(text, str):
            text = [text]
        legend = True
        color = next(self.marker_color)
        marker = next(self.marker_shape)

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            self.fig.add_trace(
                go.Scatter(
                    x=[ser_plot[col + plots]],
                    y=[i],
                    mode="markers",
                    marker={
                        "color": rgb_to_color(color, opacity=0.5),
                        "size": 10,
                        "symbol": marker,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(color),
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=text,
                    customdata=[ser_plot[col + hover]],
                    name=name,
                    showlegend=legend,
                )
            )
            legend = False

            self.fig.add_annotation(
                x=0,
                y=i + 0.4,
                text=self.annotation_text.format(
                    metric_name=metric_name,
                    data=(
                        ser_plot[col]
                        # if self.plot_type == "scout"
                        # else ser_plot[col + hover]
                    ),
                ),
                showarrow=False,
                font={
                    "color": rgb_to_color(self.white),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            )

    # def add_player(self, player: Player, n_group,metrics):

    #     # Make list of all metrics with _Z and _Rank added at end
    #     metrics_Z = [metric + "_Z" for metric in metrics]
    #     metrics_Ranks = [metric + "_Ranks" for metric in metrics]

    #     self.add_data_point(
    #         ser_plot=player.ser_metrics,
    #         plots = '_Z',
    #         name=player.name,
    #         hover='_Ranks',
    #         hover_string="Rank: %{customdata}/" + str(n_group)
    #     )

    def add_player(self, player: Union[Player, Country], n_group, metrics):

        # # Make list of all metrics with _Z and _Rank added at end
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        # Determine the appropriate attributes for player or country
        if isinstance(player, Player):
            ser_plot = player.ser_metrics
            name = player.name
        elif isinstance(player, Country):  # Adjust this based on your class structure
            ser_plot = (
                player.ser_metrics
            )  # Assuming countries have a similar metric structure
            name = player.name
        else:
            raise TypeError("Invalid player type: expected Player or Country")

        self.add_data_point(
            ser_plot=ser_plot,
            plots="_Z",
            name=name,
            hover="_Ranks",
            hover_string="Rank: %{customdata}/" + str(n_group),
        )

    # def add_players(self, players: PlayerStats, metrics):

    #     # Make list of all metrics with _Z and _Rank added at end
    #     metrics_Z = [metric + "_Z" for metric in metrics]
    #     metrics_Ranks = [metric + "_Ranks" for metric in metrics]

    #     self.add_group_data(
    #         df_plot=players.df,
    #         plots="_Z",
    #         names=players.df["player_name"],
    #         hover="_Ranks",
    #         hover_string="Rank: %{customdata}/" + str(len(players.df)),
    #         legend=f"Other players  ",  # space at end is important
    #     )

    def add_players(self, players: Union[PlayerStats, CountryStats], metrics):

        # Make list of all metrics with _Z and _Rank added at end
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        if isinstance(players, PlayerStats):
            self.add_group_data(
                df_plot=players.df,
                plots="_Z",
                names=players.df["player_name"],
                hover="_Ranks",
                hover_string="Rank: %{customdata}/" + str(len(players.df)),
                legend=f"Other players  ",  # space at end is important
            )
        elif isinstance(players, CountryStats):
            self.add_group_data(
                df_plot=players.df,
                plots="_Z",
                names=players.df["country"],
                hover="_Ranks",
                hover_string="Rank: %{customdata}/" + str(len(players.df)),
                legend=f"Other countries  ",  # space at end is important
            )
        else:
            raise TypeError("Invalid player type: expected Player or Country")

    # def add_title_from_player(self, player: Player):
    #     self.player = player

    #     title = f"Evaluation of {player.name}?"
    #     subtitle = f"Based on {player.minutes_played} minutes played"

    #     self.add_title(title, subtitle)

    def add_title_from_player(self, player: Union[Player, Country]):
        self.player = player

        title = f"Evaluation of {player.name}?"
        if isinstance(player, Player):
            subtitle = f"Based on {player.minutes_played} minutes played"
        elif isinstance(player, Country):
            subtitle = f"Based on questions answered in the World Values Survey"
        else:
            raise TypeError("Invalid player type: expected Player or Country")

        self.add_title(title, subtitle)


# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------


class DistributionPlotPersonality(Visual):
    def __init__(self, columns, *args, **kwargs):
        self.empty = True
        self.columns = columns
        self.marker_color = (
            c for c in [Visual.white, Visual.bright_yellow, Visual.bright_blue]
        )
        self.marker_shape = (s for s in ["square", "hexagon", "diamond"])
        super().__init__(*args, **kwargs)
        self._setup_axes()

    def _setup_axes(self):
        self.fig.update_xaxes(
            range=[-4, 4],
            fixedrange=True,
            tickmode="array",
            tickvals=[-3, 0, 3],
            ticktext=["Worse", "Average", "Better"],
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
        )

    def add_group_data(self, df_plot, plots, names, legend, hover="", hover_string=""):
        showlegend = True

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            temp_df = pd.DataFrame(df_plot[col + hover])
            temp_df["name"] = metric_name

            self.fig.add_trace(
                go.Scatter(
                    x=df_plot[col + plots],
                    y=np.ones(len(df_plot)) * i,
                    mode="markers",
                    marker={
                        "color": rgb_to_color(self.bright_green, opacity=0.2),
                        "size": 10,
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=names,
                    customdata=round(df_plot[col + hover]),
                    name=legend,
                    showlegend=showlegend,
                )
            )
            showlegend = False

    def add_data_point(
        self, ser_plot, plots, name, hover="", hover_string="", text=None
    ):
        if text is None:
            text = [name]
        elif isinstance(text, str):
            text = [text]
        legend = True
        color = next(self.marker_color)
        marker = next(self.marker_shape)

        for i, col in enumerate(self.columns):
            temp_hover_string = hover_string

            metric_name = format_metric(col)

            self.fig.add_trace(
                go.Scatter(
                    x=[ser_plot[col + plots]],
                    y=[i],
                    mode="markers",
                    marker={
                        "color": rgb_to_color(color, opacity=0.5),
                        "size": 10,
                        "symbol": marker,
                        "line_width": 1.5,
                        "line_color": rgb_to_color(color),
                    },
                    hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
                    text=text,
                    customdata=[round(ser_plot[col + hover])],
                    name=name,
                    showlegend=legend,
                )
            )
            legend = False

            self.fig.add_annotation(
                x=0,
                y=i + 0.4,
                text=f"<span style=''>{metric_name}: {int(ser_plot[col]):.0f}</span>",
                showarrow=False,
                font={
                    "color": rgb_to_color(self.white),
                    "family": "Gilroy-Light",
                    "size": 12 * self.font_size_multiplier,
                },
            )

    def add_person(self, person: Person, n_group, metrics):
        # Make list of all metrics with _Z and _Rank added at end
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        self.add_data_point(
            ser_plot=person.ser_metrics,
            plots="_Z",
            name=person.name,
            hover="_Ranks",
            hover_string="Rank: %{customdata}/" + str(n_group),
        )

    def add_persons(self, persons: PersonStat, metrics):

        # Make list of all metrics with _Z and _Rank added at end
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        self.add_group_data(
            df_plot=persons.df,
            plots="_Z",
            names=persons.df["name"],
            hover="_Ranks",
            hover_string="Rank: %{customdata}/" + str(len(persons.df)),
            legend=f"Other persons  ",
        )

    def add_title_from_person(self, person: Person):
        self.person = person
        title = f"Evaluation of {person.name}"
        subtitle = f"Based on Big Five scores"
        self.add_title(title, subtitle)



class ShotsPlot(Visual):
    

    def __init__(self, *args, **kwargs):
        self.empty = True
        font_path = 'data/ressources/fonts/Poppins-Regular.ttf'
        self.font_props = font_manager.FontProperties(fname=font_path)

        self.background_color = '#F5F5F5'
        self.primary_text_color = '#000000'
        self.secondary_text_color = '#757575'
        self.primary_color = '#649CCB'
        super().__init__(*args, **kwargs)
        self.fig = plt.figure(figsize=(8,12))
        self.fig.patch.set_facecolor(self.background_color)
        self.fig.patch.set_linewidth(1)
        self.fig.patch.set_edgecolor(self.secondary_text_color)
        # if labels is not None:
        #     self._setup_axes(labels)
        # else:
        #     self._setup_axes()
    #     # self._setup_pitch()



    def _setup_axes(self, labels=["Worse", "Average", "Better"]):
        self.fig.update_xaxes(
            range=[-4, 4],
            fixedrange=True,
            tickmode="array",
            tickvals=[-3, 0, 3],
            ticktext=labels,
        )
        self.fig.update_yaxes(
            showticklabels=False,
            fixedrange=True,
            gridcolor=rgb_to_color(self.medium_green),
            zerolinecolor=rgb_to_color(self.medium_green),
        )

    def add_group_data(self, df_plot, plots, names, legend, hover="", hover_string=""):
        "showlegend = True"

        # for i, col in enumerate(self.columns):
        #     temp_hover_string = hover_string

        #     metric_name = format_metric(col)

        #     temp_df = pd.DataFrame(df_plot[col + hover])
        #     temp_df["name"] = metric_name

        #     self.fig.add_trace(
        #         go.Scatter(
        #             x=df_plot[col + plots],
        #             y=np.ones(len(df_plot)) * i,
        #             mode="markers",
        #             marker={
        #                 "color": rgb_to_color(self.bright_green, opacity=0.2),
        #                 "size": 10,
        #             },
        #             hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
        #             text=names,
        #             customdata=df_plot[col + hover],
        #             name=legend,
        #             showlegend=showlegend,
        #         )
        #     )
        #     showlegend = False

    def add_data_point(self, name, player_shots):
        ""
        # legend = True
        # xg_bins = np.arange(0, player_shots['shot_statsbomb_xg'].max() + 0.1, 0.1)
        # cmap = create_pastel_cmap("Blues", n_colors=len(xg_bins), blend_ratio=0.3)

        # for _, row in player_shots.iterrows():
        #     marker = get_marker(row.sub_type_name, row.body_part_name)
        #     rounded_xg = round(row.shot_statsbomb_xg, 1)
        #     color = cmap(int(rounded_xg * 10))
        #     edgecolor=hex_to_rgb('#3473ad')
        #     linewidth=1
        #     alpha=1
        #     st.write(rgb_to_color(color, opacity=alpha))
        #     if row.outcome_name == 'Goal' :
        #         self.fig.add_trace(
        #         go.Scatter(x=[row.y],
        #                     y=[row.x],
        #                    mode="markers",
        #                    marker={
        #                         "color": 'rgba(0,0,0,0)',
        #                         "size": 30,
        #                         "symbol": marker,
        #                         "line_width": linewidth,
        #                         "line_color": edgecolor,
        #                     },
        #         ))
        #     elif row.outcome_name == 'Saved to Post' or row.outcome_name == 'Saved':
        #         edgecolor = hex_to_rgb('#000000')
        #         linewidth=1.5
        #     elif row.outcome_name == 'Off T' or row.outcome_name == 'Wayward' or row.outcome_name == 'Post':
        #         edgecolor = hex_to_rgb('#000000')
        #         linewidth=0.8
        #         color=hex_to_rgb('#a4a8b0')
        #         alpha=0.2
                
        #     elif row.outcome_name == 'Blocked':
        #         edgecolor = hex_to_rgb('#a4a8b0')
        #         linewidth=0.8
        #         # color = 'gray'
        #     self.fig.add_trace(
        #         go.Scatter(x=[row.y],
        #                     y=[row.x],
        #                    mode="markers",
        #                    marker={
        #                         "color": rounded_xg,
        #                         "colorscale":'Blues',
        #                         "size": 20,
        #                         "symbol": marker,
        #                         "line_width": linewidth,
        #                         "line_color": edgecolor,
        #                     },
        #         ))


        #     # pitch.scatter(row.x, row.y,
        #     #                     # size varies between 100 and 1900 (points squared)
        #     #                     s=100,
        #     #                     alpha=alpha,# give the markers a charcoal border
        #     #                 edgecolor=edgecolor,
        #     #                     c=color,
        #     #                 linewidths=linewidth,
        #     #                     # c='#3473ad',  # color for scatter in hex format
        #     #                     # for other markers types see: https://matplotlib.org/api/markers_api.html
        #     #                     marker=marker,
        #     #                     ax=ax)

        # #     self.fig.add_trace(
        # #         go.Scatter(
        # #             x=[ser_plot[col + plots]],
        # #             y=[i],
        # #             mode="markers",
        # #             marker={
        # #                 "color": rgb_to_color(color, opacity=0.5),
        # #                 "size": 10,
        # #                 "symbol": marker,
        # #                 "line_width": 1.5,
        # #                 "line_color": rgb_to_color(color),
        # #             },
        # #             hovertemplate="%{text}<br>" + temp_hover_string + "<extra></extra>",
        # #             text=text,
        # #             customdata=[ser_plot[col + hover]],
        # #             name=name,
        # #             showlegend=legend,
        # #         )
        # #     )
        # #     legend = False

        # #     self.fig.add_annotation(
        # #         x=0,
        # #         y=i + 0.4,
        # #         text=self.annotation_text.format(
        # #             metric_name=metric_name,
        # #             data=(
        # #                 ser_plot[col]
        # #                 # if self.plot_type == "scout"
        # #                 # else ser_plot[col + hover]
        # #             ),
        # #         ),
        # #         showarrow=False,
        # #         font={
        # #             "color": rgb_to_color(self.white),
        # #             "family": "Gilroy-Light",
        # #             "size": 12 * self.font_size_multiplier,
        # #         },
        # #     )

    def plot_shots(self, player_shots, stats):
        ax2 = self.fig.add_axes([.05, .3,.91, .5])
        ax2.set_facecolor(self.background_color)

        pitch = VerticalPitch(pitch_type='statsbomb',  
                            line_zorder=0, 
                            line_color=self.secondary_text_color, 
                            half=True, 
                            pitch_color=self.background_color, 
                            pad_bottom=.5,
                            linewidth=.75,
                            axis=True,
                            label=True,
                            corner_arcs=True,
                            goal_type='box')
        pitch.draw(ax=ax2)

        ax2.scatter(x=7, y=stats['avg_distance_location'], color=self.secondary_text_color,linewidth=.5, s=50)
        ax2.plot([7,7], [stats['avg_distance_location'],120], color=self.secondary_text_color, linewidth=1)
        ax2.text(x=7, y=stats['avg_distance_location']-4, s=f'Avg. Distance\n{stats["avg_distance"]:.1f} meters',
                fontsize=8, fontproperties=self.font_props, color=self.secondary_text_color, ha='center')

        for shot in player_shots.to_dict(orient='records'):
            pitch.scatter(shot['x'], shot['y'],
                            s=300 * shot['shot_statsbomb_xg'],
                            color=self.primary_color if shot['outcome_name'] == 'Goal' else self.background_color,
                            alpha=.7,
                            edgecolor=self.secondary_text_color, linewidth=.8,ax=ax2)
        
        ax2.axis('off')

    def add_player_shots(self, playerShots: PlayerShots):

        # # # Make list of all metrics with _Z and _Rank added at end
        # metrics_Z = [metric + "_Z" for metric in metrics]
        # metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        # Determine the appropriate attributes for player or country
        if isinstance(playerShots, PlayerShots):
            ser_plot = playerShots.player_shots
            name = playerShots.name
            stats = playerShots.ser_metrics
        else:
            raise TypeError("Invalid player type: expected Player or Country")

        self.plot_shots(
            player_shots=ser_plot,
            stats=stats
        )

    def add_stats(self, playerShots: PlayerShots):
        stats = playerShots.ser_metrics

        ax3 = self.fig.add_axes([0, .25, 1, .05])
        ax3.set_facecolor(self.background_color)

        ax3.text(x=.2, y=.5, s='Shots', fontsize=18, fontproperties=self.font_props, fontweight='bold', color=self.secondary_text_color, ha='center')
        ax3.text(x=.2, y=.1, s=f'{stats["total_shots"]}', fontsize=14, fontproperties=self.font_props, color=self.primary_text_color, ha='center')

        ax3.text(x=.4, y=.5, s='Goals', fontsize=18, fontproperties=self.font_props, fontweight='bold', color=self.secondary_text_color, ha='center')
        ax3.text(x=.4, y=.1, s=f'{stats["goals"]}', fontsize=14, fontproperties=self.font_props, color=self.primary_text_color, ha='center')

        ax3.text(x=.6, y=.5, s=' xG', fontsize=18, fontproperties=self.font_props, fontweight='bold', color=self.secondary_text_color, ha='center')
        ax3.text(x=.6, y=.1, s=f'{stats["total_xG"]:.2f}', fontsize=14, fontproperties=self.font_props, color=self.primary_text_color, ha='center')

        ax3.text(x=.8, y=.5, s='xG/shot', fontsize=18, fontproperties=self.font_props, fontweight='bold', color=self.secondary_text_color, ha='center')
        ax3.text(x=.8, y=0, s=f'{stats["xG_per_shot"]:.2f}', fontsize=14, fontproperties=self.font_props, color=self.primary_text_color, ha='center')
        
        ax3.axis('off')

    def add_players(self, players: Union[PlayerStats, CountryStats], metrics):

        # Make list of all metrics with _Z and _Rank added at end
        metrics_Z = [metric + "_Z" for metric in metrics]
        metrics_Ranks = [metric + "_Ranks" for metric in metrics]

        if isinstance(players, PlayerStats):
            self.add_group_data(
                df_plot=players.df,
                plots="_Z",
                names=players.df["player_name"],
                hover="_Ranks",
                hover_string="Rank: %{customdata}/" + str(len(players.df)),
                legend=f"Other players  ",  # space at end is important
            )
        elif isinstance(players, CountryStats):
            self.add_group_data(
                df_plot=players.df,
                plots="_Z",
                names=players.df["country"],
                hover="_Ranks",
                hover_string="Rank: %{customdata}/" + str(len(players.df)),
                legend=f"Other countries  ",  # space at end is important
            )
        else:
            raise TypeError("Invalid player type: expected Player or Country")

    def add_title_from_player(self, player_shots: PlayerShots):
        player = player_shots.name
        ax1 = self.fig.add_axes([0, .7, 1, .2])
        ax1.set_facecolor(self.background_color)
        ax1.set_xlim(0,1)
        ax1.set_ylim(0,1)

        ax1.text(x=.5, y=.9, s=player, fontsize=20, fontproperties=self.font_props, fontweight='bold', color=self.primary_text_color, ha='center')
        ax1.text(x=.5, y=.8, s='All non-penalty shots in the Copa América 2024', fontsize=14, fontproperties=self.font_props, fontweight='bold', color=self.primary_text_color, ha='center')

        ax1.text(x=.41, y=.61, s='Low xG', fontsize=12, fontproperties=self.font_props, fontweight='bold', color=self.secondary_text_color, ha='right')
        ax1.scatter(x=.43, y=.63, s=100, color=self.background_color, edgecolor=self.secondary_text_color, linewidth=.8)
        ax1.scatter(x=.46, y=.63, s=200, color=self.background_color, edgecolor=self.secondary_text_color, linewidth=.8)
        ax1.scatter(x=.5, y=.63, s=300, color=self.background_color, edgecolor=self.secondary_text_color, linewidth=.8)
        ax1.scatter(x=.545, y=.63, s=400, color=self.background_color, edgecolor=self.secondary_text_color, linewidth=.8)
        ax1.scatter(x=.595, y=.63, s=500, color=self.background_color, edgecolor=self.secondary_text_color, linewidth=.8)
        ax1.text(x=.625, y=.61, s='High xG', fontsize=12, fontproperties=self.font_props, fontweight='bold', color=self.secondary_text_color, ha='left')

        ax1.text(x=.47, y=.43, s='Goal', fontsize=10, fontproperties=self.font_props, fontweight='bold', color=self.secondary_text_color, ha='right')
        ax1.scatter(x=.485, y=.45, s=100, color=self.primary_color, edgecolor=self.secondary_text_color, linewidth=.8, alpha=.7)
        ax1.scatter(x=.525, y=.45, s=100, color=self.background_color, edgecolor=self.secondary_text_color, linewidth=.8)
        ax1.text(x=.54, y=.43, s='No Goal', fontsize=10, fontproperties=self.font_props, fontweight='bold', color=self.secondary_text_color, ha='left')
        ax1.axis('off')
        
    
    def add_footer(self):
        ax4 = self.fig.add_axes([0, .15, 1, .05])
        ax4.set_facecolor(self.background_color)
        ax4.set_xlim(-.3,1)
        ax4.set_ylim(-.3,1)

        ax4.text(x=-.3, y=.06, s='Data from', fontsize=8, fontproperties=self.font_props, fontweight='regular', color=self.secondary_text_color, ha='left')
        img = Image.open('data/ressources/img/statsbomb_logo.png')
        ax4.text(x=1, y=-.3, s='Martín Steglich', fontsize=8, fontproperties=self.font_props, fontweight='regular', color=self.secondary_text_color, ha='right')
        image_position = [-.3, -.13, -.3, 0]  # [x_min, x_max, y_min, y_max]
        ax4.imshow(img, extent=image_position, aspect='auto', zorder=2)

        ax4.axis('off')

    def show_plot(self):
        st.pyplot(self.fig)