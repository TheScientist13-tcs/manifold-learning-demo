import plotly.graph_objects as go
import pandas as pd
import plotly.express as px


def grouped_scatterplot(df, x_name, y_name, grouping_name, marker_size=3):
    # Step 3: Get unique groups and assign colors from Plotly's qualitative color scale

    df["group"] = list(map(str, df[grouping_name]))
    unique_groups = df["group"].unique()
    color_sequence = px.colors.qualitative.Plotly  # Use default Plotly color sequence

    # Create a mapping of groups to colors
    color_map = {
        group: color_sequence[i % len(color_sequence)]
        for i, group in enumerate(unique_groups)
    }

    # Step 4: Create the figure
    fig = go.Figure()

    # Add a separate trace for each group to ensure legends are displayed
    for group in unique_groups:
        group_data = df[df["group"] == group]
        fig.add_trace(
            go.Scatter(
                x=group_data[x_name],
                y=group_data[y_name],
                mode="markers",
                marker=dict(size=marker_size, color=color_map[group]),
                name=group,  # Set the name for the legend
            )
        )

    # Step 5: Update layout and display the plot
    fig.update_layout(
        legend_title="Legend",
    )

    return fig


def grouped_3dscatterplot(df, x_name, y_name, z_name, grouping_name, marker_size=3):
    # Step 3: Get unique groups and assign colors from Plotly's qualitative color scale

    df["group"] = list(map(str, df[grouping_name]))
    unique_groups = df["group"].unique()
    color_sequence = px.colors.qualitative.Plotly  # Use default Plotly color sequence

    # Create a mapping of groups to colors
    color_map = {
        group: color_sequence[i % len(color_sequence)]
        for i, group in enumerate(unique_groups)
    }

    # Step 4: Create the figure
    fig = go.Figure()

    # Add a separate trace for each group to ensure legends are displayed
    for group in unique_groups:
        group_data = df[df["group"] == group]
        fig.add_trace(
            go.Scatter3d(
                x=group_data[x_name],
                y=group_data[y_name],
                z=group_data[z_name],
                mode="markers",
                marker=dict(size=marker_size, color=color_map[group]),
                name=group,  # Set the name for the legend
            )
        )

    # Step 5: Update layout and display the plot
    fig.update_layout(
        legend_title="Legend",
    )

    return fig
