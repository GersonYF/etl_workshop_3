import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from io import StringIO
from wordcloud import WordCloud

colors = sns.color_palette([
    "#008000",  # Original Green
    "#ffd700",  # Original Gold
    "#66cdaa",  # Lighter Green
    "#ccac00",  # Darker Gold
    "#003366",  # Dark Midnight Blue
    "#f5f5dc",  # Beige
    "#809b8d"   # Cadet Grey
])

start_bold = "\033[1m"
end_bold = "\033[0;0m"

def clean_column_names(columns) -> list[str]:
    """
    Clean column names by converting them to lower case and replacing spaces and hyphens with underscores.
        :param columns: list of column names
        :return: list of cleaned column names

    """
    return [col.lower().replace(" ", "_").replace("-", "_") for col in columns]


def get_color(value, high_threshold = 95, low_threshold = 40) -> str:
    """
    Get color based on value and thresholds.
        :param value: value to be colored
        :param high_threshold: high threshold for red color
        :param low_threshold: low threshold for blue color
        :return: color string
    """
    if value > high_threshold:
        return colors[len(colors) - 1]
    if value > low_threshold:
        return colors[math.floor((len(colors)-1) / 2)]

    return colors[0]


def plot_counts(value_counts, high_threshold, low_threshold) -> None:
    """
    Plot value counts with color coding based on thresholds.
        :param value_counts: value counts series
        :param high_threshold: high threshold for red color
        :param low_threshold: low threshold for blue color
    """
    sorted_counts = value_counts.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_counts.index, sorted_counts, color=colors[len(colors) - 1])
    plt.xlabel('Columns')
    plt.ylabel('Counts')
    plt.title('Counts in Each Column')
    plt.xticks(rotation=90)
    plt.grid(which='both', linestyle='--', linewidth=0.5, zorder=0)

    for bar in bars:
        yval = bar.get_height()
        color = get_color(yval, high_threshold, low_threshold)
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{round(yval, 2)}", va='bottom', color=color)

    plt.show()


def plot_null_counts(df, high_threshold, low_threshold) -> None:
    """
    Plot null counts with color coding based on thresholds.
        :param df: dataframe
        :param high_threshold: high threshold for red color
        :param low_threshold: low threshold for blue color
    """
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    null_percentages = null_counts / df.shape[0] * 100
    sorted_null_percentages = null_percentages.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_null_percentages.index, sorted_null_percentages)
    plt.xlabel('Columns')
    plt.ylabel('Percentage of Nulls')
    plt.title('Percentage of Nulls in Each Column')
    plt.xticks(rotation=90)
    plt.grid(which='both', linestyle='--', linewidth=0.5, zorder=0)

    for bar in bars:
        yval = bar.get_height()
        color = get_color(yval, high_threshold, low_threshold)
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{round(yval, 2)}%", va='bottom', color=color)

    plt.show()


def plot_with_capped_limit(df, column, quantiles, ylabel, plot_type='box') -> None:
    """
    Plot original and capped data for a column.
        :param df: dataframe
        :param column: column to be plotted
        :param quantiles: list of quantiles to cap the data
        :param ylabel: label for y-axis
        :param plot_type: type of plot, box or hist
    """
    if plot_type not in ['box', 'hist']:
        raise ValueError('plot_type should be one of "box" or "hist"')

    if not quantiles:
        getattr(df[column].plot, plot_type)()
        plt.title('Original')
        plt.ylabel(ylabel)

        if plot_type == 'hist':
            df[column].plot.density()

        plt.grid(which='both', linestyle='--', linewidth=0.5, zorder=0)
        plt.tight_layout()
        plt.show()
        return

    _, axs = plt.subplots(1, len(quantiles) + 1, figsize=(5 * (len(quantiles) + 1), 5))

    getattr(df[column].plot, plot_type)(ax=axs[0])
    axs[0].set_title('Original')
    axs[0].set_ylabel(ylabel)

    if plot_type == 'hist':
        df[column].plot.density(ax=axs[0])

    for i, quantile in enumerate(quantiles, start=1):
        cap = df[column].quantile(quantile)
        capped_data = df[column].where(df[column] <= cap, cap)
        getattr(capped_data.plot, plot_type)(ax=axs[i])

        if plot_type == 'hist':
            capped_data.plot.density(ax=axs[i])

        axs[i].set_title(f'Capped at {quantile} quantile')

    plt.grid(which='both', linestyle='--', linewidth=0.5, zorder=0)
    plt.tight_layout()
    plt.show()


def plot_stacked_bar_from_combinations(df, columns) -> None:
    """
    Plot stacked bar chart from combinations of columns.
        :param df: dataframe
        :param columns: list of columns
    """
    value_counts = df[columns].value_counts()
    counts_df = value_counts.reset_index()
    counts_df.columns = columns + ['Count']

    counts_df['Label'] = counts_df[columns].apply(lambda row: ', '.join(row.index[row].tolist()), axis=1)

    plt.figure(figsize=(10, 6))
    bottom_pos = pd.Series([0] * counts_df.shape[0])
    for col in columns:
        counts = counts_df['Count'].where(counts_df[col]).fillna(0)
        plt.bar(counts_df['Label'], counts, bottom=bottom_pos, label=col)
        bottom_pos += counts

    plt.xlabel('OS Combination')
    plt.ylabel('Count')
    plt.title('Stacked Bar Chart of Value Counts for Combinations')
    plt.xticks(rotation=45)

    plt.grid(which='both', linestyle='--', linewidth=0.5, zorder=0)

    plt.legend()
    plt.tight_layout()
    plt.show()


def generate_text_cloud(df, column) -> str:
    """
    Generate text cloud from a column.
        :param df: dataframe
        :param column: column to generate text cloud
        :return: string cloud
    """
    si = StringIO()
    df[column].apply(lambda x: si.write(str(x)))
    string_cloud = si.getvalue()
    si.close()

    return string_cloud


def plot_word_cloud(df, column) -> None:
    """
    Plot word cloud from a column.
        :param df: dataframe
        :param column: column to generate word cloud
    """
    text_cloud = generate_text_cloud(df, column)

    wordcloud = WordCloud(
        background_color="white",
        max_words=len(text_cloud),
        max_font_size=40,
        relative_scaling=.5
    ).generate(text_cloud)

    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

def get_min_max_counts(value_counts) -> tuple:
    """
    Get index of minimum and maximum counts from value counts.
        :param value_counts: value counts series
        :return: tuple of index of minimum and maximum counts
    """
    years_with_min_applications = value_counts[value_counts == value_counts.min()].index.tolist()[0]
    years_with_max_applications = value_counts[value_counts == value_counts.max()].index.tolist()[0]
    
    return years_with_min_applications, years_with_max_applications


def replace_pattern_in_file(data_path, pattern, replacement):
    """
    Replace a specific pattern with a replacement character in a file.
    
    :param data_path: The path to the file
    :param pattern: The regex pattern to replace
    :param replacement: The character to replace the pattern with
    """
    with open(data_path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    file_content = re.sub(pattern, replacement, file_content)

    with open(data_path, 'w', encoding='utf8') as file:
        file.write(file_content)


def get_min_count_values(df, column):
    """
    Get the values that appear the minimum number of times in the dataset.
    df: pd.DataFrame
    column: str
    """
    unique_values = df[column].unique()
    value_counts = df[column].value_counts()
    min_count = value_counts.min()
    
    min_count_values = [value for value in unique_values if value_counts[value] == min_count]
    
    return min_count_values


def get_max_count_values(df, column):
    """
    Get the values that appear the maximum number of times in the dataset.
    df: pd.DataFrame
    column: str
    """
    unique_values = df[column].unique()
    value_counts = df[column].value_counts()
    max_count = value_counts.max()
    
    max_count_values = [value for value in unique_values if value_counts[value] == max_count]
    
    return max_count_values