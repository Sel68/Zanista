import matplotlib.pyplot as plt
import seaborn as sns #used for ease in visuals; built on matplot
import pandas as pd
import numpy as np
import streamlit as st  # For steamlit warnings


#Applies title and labels to a Matplotlib Axes object.
def _apply_common_layout_matplotlib(ax, title, x_label=None, y_label=None):
    if not ax:
        return None
    ax.set_title(title, fontweight="bold")
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    plt.tight_layout()
    
    return ax.figure  # Return the figure containing the axes


#Creates plots using Matplotlib/Seaborn.
def create_visualizations(df, viz_type, x_column=None, y_column=None, color_column=None):

    fig, ax = plt.subplots()  #figure and axes objects
    title = f"{viz_type}"
    x_label, y_label = x_column, y_column

    # Basic check for X column
    if not x_column:
        plt.close(fig)  # Close the unused figure
        raise ValueError("X column is required for visualization.")

    sns.set_theme(style="whitegrid")

    try:
        if viz_type == "Scatter Plot":
            if not y_column:
                plt.close(fig)
                raise ValueError("Y column needed for Scatter Plot.")
            if not pd.api.types.is_numeric_dtype(df.get(x_column)) or not pd.api.types.is_numeric_dtype(df.get(y_column)):
                
                plt.close(fig)
                raise ValueError("Scatter plot requires numeric X and Y columns.")

            title = f"{y_column} vs {x_column}" + (
                f" by {color_column}" if color_column else ""
            )
            sns.scatterplot(
                data=df,
                x=x_column,
                y=y_column,
                color="skyblue", #preset
                edgecolor="none",
                hue=color_column,
                ax=ax,
                palette="viridis",
            )

        elif viz_type == "Histogram":
            if not pd.api.types.is_numeric_dtype(df.get(x_column)):
                plt.close(fig)
                raise ValueError("Histogram requires a numeric column.")

            title = f"Distribution of {x_column}" + (
                f" by {color_column}" if color_column else ""
            )
            y_label = "Frequency"  #standard label for histogram

            if color_column:
                unique_colors = df[color_column].dropna().unique()
                data_list = [
                    df[df[color_column] == val][x_column] for val in unique_colors
                ]
                ax.hist(
                    data_list,
                    bins=30,
                    stacked=True,
                    color=plt.cm.viridis(range(len(unique_colors))),
                    edgecolor="black",
                )
                ax.legend(unique_colors)
            else:
                ax.hist(df[x_column], bins=30, color="skyblue", edgecolor="black")

            ax.set_title(title)
            ax.set_ylabel(y_label)
            ax.set_xlabel(x_column)

        elif viz_type == "Bar Chart":
            title = f"Bar Chart for {x_column}"
            # If Y is not numeric, we defaulted it to do counterplot
            if not y_column or not pd.api.types.is_numeric_dtype(df.get(y_column)):
                y_label = "Count"
                title = f"Counts by {x_column}" + (
                    f" (colored by {color_column})"
                    if color_column and color_column != x_column
                    else ""
                )
                # x_column for x-axis
                hue_col = (
                    color_column if color_column and color_column != x_column else None
                )
                sns.countplot(
                    data=df, x=x_column, hue=hue_col, ax=ax, palette="viridis"
                )
            else:  # if Ys numeric, calculate mean and plot barplot
                y_label = f"Mean of {y_column}"
                title = f"Mean {y_column} by {x_column}" + (
                    f" (colored by {color_column})"
                    if color_column and color_column != x_column
                    else ""
                )
                # Grouping for mean calculation is handled by seaborn's barplot estimator
                hue_col = (
                    color_column if color_column and color_column != x_column else None
                )
                sns.barplot(
                    data=df,
                    x=x_column,
                    y=y_column,
                    hue=hue_col,
                    ax=ax,
                    estimator=np.mean,
                    errorbar=None,
                    palette="viridis",
                ) 

            ax.tick_params(axis="x", rotation=45)

        elif viz_type == "Pie Chart":
            title = f"Distribution: {x_column}"
            pie_data = None
            labels = None
            use_counts = True

            if y_column and pd.api.types.is_numeric_dtype(df.get(y_column)):
                try:
                    # x_column and sum y_column grouping; take top N if too many
                    grouped_data = (
                        df.groupby(x_column, observed=False)[y_column]
                        .sum()
                        .reset_index()
                    )

                    # Limit to top 10 categories for pie chart
                    grouped_data = grouped_data.nlargest(10, y_column) 
                    if not grouped_data.empty:
                        pie_data = grouped_data[y_column]
                        labels = grouped_data[x_column]
                        title = f"Sum of {y_column} by {x_column}"
                        use_counts = False
                    else:
                        st.warning(
                            f"Could not calculate sum of '{y_column}' for pie chart. Showing counts instead."
                        )
                except Exception:  # Fallback to counts on error
                    st.warning(
                        f"Error calculating sum of '{y_column}'. Showing counts instead."
                    )
                    use_counts = True

            if use_counts:
                # Use value_counts for counts. Limit to top 10.
                counts = df[x_column].value_counts().nlargest(10)
                if not counts.empty:
                    pie_data = counts.values
                    labels = counts.index
                    title = f"Counts by {x_column}"
                else:
                    plt.close(fig)
                    raise ValueError(
                        f"No data available for column '{x_column}' to create a pie chart."
                    )

            # create pi
            ax.pie(
                pie_data,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops=dict(width=0.4),
            )  # Donut style lol (its empty in the middle)
            ax.axis("equal")  # equal aspect ratio so circular pi
            y_label = None  # no Y-label for pi

        else:
            plt.close(fig)  # Close figure if type is unknown
            raise ValueError(f"Unsupported visualization type: {viz_type}")

        # Apply common layout elements
        fig = _apply_common_layout_matplotlib(ax, title, x_label, y_label)

    except Exception as e:
        plt.close(fig)  # Ensure figure is closed on error
        st.error(f"Failed to create plot: {e}")
        return None  # Return None on error

    return fig

#Creates a Whisker
def create_whisker_plot(series):

    #only for numeric data
    if not pd.api.types.is_numeric_dtype(series):
        st.warning(f"Column '{series.name}' is not numeric. Cannot create box plot.")
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.set_theme(style="whitegrid")

    try:
        sns.boxplot(y=series, ax=ax, palette="viridis")
        #apply comm layout
        fig = _apply_common_layout_matplotlib(ax, title=f"Box Plot: {series.name}", y_label=series.name)

    except Exception as e:
        plt.close(fig) 
        st.error(f"Failed to create box plot for '{series.name}': {e}")
        return None

    return fig
