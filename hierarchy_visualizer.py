from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import (
    Dash,
    Input,
    Output,
    State,
    callback,
    callback_context,
    dash_table,
    dcc,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate


CSV_PATH = Path(__file__).with_name("nodes.csv")


def load_nodes(path: Path) -> tuple[pd.DataFrame, dict[int, dict], int]:
    """Load and validate the hierarchy CSV."""
    df = pd.read_csv(path)

    required = {"id", "label", "description", "parent", "children"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}"
        )

    df["id"] = df["id"].astype(int)
    df["parent"] = df["parent"].astype(int)
    df["label"] = df["label"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    def parse_children(value) -> list[int]:
        if pd.isna(value) or str(value).strip() == "":
            return []

        parsed = ast.literal_eval(str(value))

        if not isinstance(parsed, list):
            raise ValueError(
                f"children must be a list, received: {value!r}"
            )

        return [int(child_id) for child_id in parsed]

    df["children_list"] = df["children"].apply(parse_children)

    if df["id"].duplicated().any():
        duplicates = df.loc[
            df["id"].duplicated(),
            "id",
        ].tolist()

        raise ValueError(
            f"Duplicate node IDs found: {duplicates[:10]}"
        )

    records = df.set_index("id").to_dict("index")

    roots = df.loc[df["parent"] == -1, "id"].tolist()

    if len(roots) != 1:
        raise ValueError(
            f"Expected exactly one root node, found {len(roots)}"
        )

    unknown_children = {
        child_id
        for children in df["children_list"]
        for child_id in children
        if child_id not in records
    }

    if unknown_children:
        raise ValueError(
            "Children reference unknown node IDs: "
            f"{sorted(unknown_children)[:10]}"
        )

    return df, records, roots[0]


nodes_df, NODES, ROOT_ID = load_nodes(CSV_PATH)

# A special UI state meaning:
# "Show the root as one bar."
ROOT_OVERVIEW = -1


@lru_cache(maxsize=None)
def leaf_count(node_id: int) -> int:
    """Return the number of leaf descendants below a node."""
    children = NODES[node_id]["children_list"]

    if not children:
        return 1

    return sum(
        leaf_count(child_id)
        for child_id in children
    )


def all_children_are_leaves(node_id: int) -> bool:
    """Return True when a node has children and every child is a leaf."""
    children = NODES[node_id]["children_list"]

    return bool(children) and all(
        not NODES[child_id]["children_list"]
        for child_id in children
    )


def node_path(node_id: int) -> list[int]:
    """Return the root-to-node path using parent IDs."""
    path = []
    current = node_id
    seen = set()

    while current != -1:
        if current in seen:
            raise ValueError("Cycle detected in parent links")

        seen.add(current)
        path.append(current)
        current = NODES[current]["parent"]

    return list(reversed(path))


def make_bar_figure(
    node_ids: list[int],
    title: str,
) -> go.Figure:
    """Create a bar chart for the supplied node IDs."""
    labels = [
        NODES[node_id]["label"]
        for node_id in node_ids
    ]

    counts = [
        leaf_count(node_id)
        for node_id in node_ids
    ]

    descriptions = [
        NODES[node_id]["description"]
        for node_id in node_ids
    ]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts,
            customdata=list(
                zip(node_ids, descriptions)
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Leaf count: %{y}<br>"
                "%{customdata[1]}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Node label",
        yaxis_title="Number of leaf nodes",
        clickmode="event+select",
        margin=dict(
            l=60,
            r=30,
            t=70,
            b=180,
        ),
        height=650,
    )

    fig.update_xaxes(
        tickangle=-45,
        automargin=True,
    )

    fig.update_yaxes(
        rangemode="tozero",
    )

    return fig


def leaf_table(node_id: int) -> dash_table.DataTable:
    """Create a table containing a node's leaf children."""
    children = NODES[node_id]["children_list"]

    rows = [
        {
            "id": child_id,
            "label": NODES[child_id]["label"],
            "description": NODES[child_id]["description"],
        }
        for child_id in children
    ]

    return dash_table.DataTable(
        data=rows,
        columns=[
            {
                "name": "Leaf ID",
                "id": "id",
            },
            {
                "name": "Label",
                "id": "label",
            },
            {
                "name": "Description",
                "id": "description",
            },
        ],
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_table={
            "overflowX": "auto",
        },
        style_cell={
            "textAlign": "left",
            "whiteSpace": "normal",
            "height": "auto",
            "padding": "10px",
            "minWidth": "120px",
            "maxWidth": "500px",
        },
        style_header={
            "fontWeight": "bold",
        },
    )


app = Dash(__name__)
app.title = "Hierarchy Explorer"


app.layout = html.Div(
    [
        dcc.Store(
            id="current-node",
            data=ROOT_OVERVIEW,
        ),

        html.H1("Hierarchy Explorer"),

        html.P(
            "Click a bar to drill into that node. Hover over a bar "
            "to see its description. When a node's children are all "
            "leaves, they appear in a table."
        ),

        html.Div(
            [
                html.Button(
                    "Back",
                    id="back-button",
                    n_clicks=0,
                    disabled=True,
                ),
                html.Button(
                    "Root",
                    id="root-button",
                    n_clicks=0,
                ),
            ],
            style={
                "display": "flex",
                "gap": "10px",
                "marginBottom": "12px",
            },
        ),

        html.Div(
            id="breadcrumb",
            style={
                "marginBottom": "12px",
                "fontWeight": "bold",
            },
        ),

        # The graph is always present in the layout.
        # It is hidden when the table is displayed.
        html.Div(
            id="chart-container",
            children=[
                dcc.Graph(
                    id="hierarchy-chart",
                    figure=make_bar_figure(
                        [ROOT_ID],
                        "Root node",
                    ),
                    config={
                        "displaylogo": False,
                        "scrollZoom": True,
                    },
                )
            ],
        ),

        # The table or leaf-node information is placed here.
        html.Div(
            id="table-container",
            style={"display": "none"},
        ),
    ],
    style={
        "maxWidth": "1400px",
        "margin": "0 auto",
        "padding": "24px",
        "fontFamily": "Arial, sans-serif",
    },
)


@callback(
    Output("current-node", "data"),
    Input("hierarchy-chart", "clickData"),
    Input("back-button", "n_clicks"),
    Input("root-button", "n_clicks"),
    State("current-node", "data"),
    prevent_initial_call=True,
)
def navigate(
    click_data,
    back_clicks,
    root_clicks,
    current_node,
):
    """Update the currently selected node."""
    triggered = callback_context.triggered_id

    if triggered == "root-button":
        return ROOT_OVERVIEW

    if triggered == "back-button":
        if current_node == ROOT_OVERVIEW:
            raise PreventUpdate

        # When viewing the root's children, Back returns
        # to the single root overview bar.
        if int(current_node) == ROOT_ID:
            return ROOT_OVERVIEW

        parent_id = NODES[int(current_node)]["parent"]

        if parent_id == -1:
            return ROOT_OVERVIEW

        return parent_id

    if triggered == "hierarchy-chart":
        if not click_data:
            raise PreventUpdate

        clicked_id = int(
            click_data["points"][0]["customdata"][0]
        )

        return clicked_id

    raise PreventUpdate


@callback(
    Output("breadcrumb", "children"),
    Output("hierarchy-chart", "figure"),
    Output("chart-container", "style"),
    Output("table-container", "children"),
    Output("table-container", "style"),
    Output("back-button", "disabled"),
    Input("current-node", "data"),
)
def render_node(current_node):
    """Render either a child-node chart or a leaf table."""

    # Initial overview: display the root as one bar.
    if current_node == ROOT_OVERVIEW:
        root = NODES[ROOT_ID]

        return (
            root["label"],
            make_bar_figure(
                [ROOT_ID],
                "Root node",
            ),
            {"display": "block"},
            None,
            {"display": "none"},
            True,
        )

    node_id = int(current_node)
    node = NODES[node_id]

    path = node_path(node_id)

    breadcrumb = "  ›  ".join(
        NODES[path_id]["label"]
        for path_id in path
    )

    # If every child is a leaf, hide the graph and show the table.
    if all_children_are_leaves(node_id):
        table_content = html.Div(
            [
                html.H2(
                    f"{node['label']} — leaf responses"
                ),
                html.P(
                    f"{leaf_count(node_id):,} leaf nodes. "
                    "Use the table filters or sorting controls "
                    "to explore them."
                ),
                leaf_table(node_id),
            ]
        )

        return (
            breadcrumb,
            no_update,
            {"display": "none"},
            table_content,
            {"display": "block"},
            False,
        )

    displayed_ids = node["children_list"]

    # Handle the unusual case where the selected node itself is a leaf.
    if not displayed_ids:
        leaf_content = html.Div(
            [
                html.H2(node["label"]),
                html.P(node["description"]),
                html.P(
                    "This node is a leaf and has no children."
                ),
            ]
        )

        return (
            breadcrumb,
            no_update,
            {"display": "none"},
            leaf_content,
            {"display": "block"},
            False,
        )

    # Otherwise, display the selected node's children.
    figure = make_bar_figure(
        displayed_ids,
        f"Children of {node['label']}",
    )

    return (
        breadcrumb,
        figure,
        {"display": "block"},
        None,
        {"display": "none"},
        False,
    )


if __name__ == "__main__":
    app.run(debug=True)
