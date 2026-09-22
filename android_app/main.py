import json
import os
import re
import threading

import requests

from kivy.app import App
from kivy.clock import Clock
from kivy.core.clipboard import Clipboard
from kivy.core.window import Window
from kivy.graphics import Color, RoundedRectangle, Line
from kivy.metrics import dp
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.utils import platform


# ============================================================
# BACKEND SERVER
# ============================================================
# This is the default backend URL baked into the APK. Replace it with
# your deployed backend's HTTPS URL (e.g. from Render/Railway) before
# building the release APK. Students can also change it later at
# runtime from the in-app Settings screen (gear icon) without needing
# a new APK, which is handy if you move hosts or are testing locally.

DEFAULT_API_BASE_URL = "https://kerala-it-hub-api.onrender.com"

# Do NOT hardcode the real API key here -- this file is committed to
# a public repo, so anything baked in is public too. Leave this blank
# and set the actual key at runtime via the in-app Settings screen
# (gear icon) instead. It must match the API_KEY environment variable
# configured on the backend (backend/main.py / your Render dashboard).
DEFAULT_API_KEY = ""

# Kivy's software-keyboard handling: slide the view up so the input
# field stays visible above the on-screen keyboard on Android.
Window.softinput_mode = "below_target"


# ============================================================
# COLOR PALETTE
# ============================================================

BG_COLOR = (0.05, 0.06, 0.08, 1)
SURFACE_COLOR = (0.11, 0.12, 0.15, 1)
BORDER_COLOR = (0.22, 0.24, 0.28, 1)
ACCENT_COLOR = (0.20, 0.70, 0.58, 1)
TEXT_COLOR = (0.93, 0.94, 0.95, 1)
MUTED_TEXT_COLOR = (0.65, 0.68, 0.72, 1)

Window.clearcolor = BG_COLOR

# Shared spacing constants so cards (course cards, source cards) share
# one consistent internal padding, and plain text/headings share a
# separate, flush baseline -- rather than the ad hoc, mismatched
# padding values each widget used to pick independently.
CARD_PADDING = dp(14)
CARD_RADIUS = dp(12)


# ============================================================
# ROUNDED BUTTON
# ============================================================
# Kivy's default Button is a flat, hard-edged rectangle. This draws a
# rounded, colored background instead so buttons match the rest of the
# UI (course cards, popups) rather than looking like a stock widget.

class RoundedButton(Button):

    def __init__(self, fill_color=ACCENT_COLOR, **kwargs):

        super().__init__(
            background_normal="",
            background_down="",
            background_color=(0, 0, 0, 0),
            color=(1, 1, 1, 1),
            **kwargs
        )

        self.fill_color = fill_color

        with self.canvas.before:
            self._color_instruction = Color(*fill_color)
            self._background = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[dp(10)]
            )

        self.bind(
            pos=self._update_background,
            size=self._update_background,
            state=self._update_state_color
        )

    def _update_background(self, *args):

        self._background.pos = self.pos
        self._background.size = self.size

    def _update_state_color(self, *args):

        if self.state == "down":
            self._color_instruction.rgba = tuple(
                max(0, channel - 0.08) for channel in self.fill_color[:3]
            ) + (self.fill_color[3],)
        else:
            self._color_instruction.rgba = self.fill_color


def get_settings_path():

    return os.path.join(
        App.get_running_app().user_data_dir,
        "settings.json"
    )


def load_settings():

    path = get_settings_path()

    if os.path.exists(path):

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return {
                "api_base_url": data.get(
                    "api_base_url",
                    DEFAULT_API_BASE_URL
                ),
                "api_key": data.get(
                    "api_key",
                    DEFAULT_API_KEY
                )
            }

        except (json.JSONDecodeError, OSError):
            pass

    return {
        "api_base_url": DEFAULT_API_BASE_URL,
        "api_key": DEFAULT_API_KEY
    }


def save_settings(api_base_url, api_key):

    path = get_settings_path()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "api_base_url": api_base_url.strip(),
                "api_key": api_key.strip()
            },
            f
        )


# ============================================================
# READ-ONLY TEXT INPUT
# ============================================================
# Kivy's TextInput schedules a "long touch" timer 0.5s after any
# touch-down, and fires the Select All/Copy bubble from it even if
# the finger never moved -- which fires constantly for any ordinary
# touch-and-hold-then-drag scroll gesture over this text, not just a
# deliberate long-press. A real drag-to-select still shows the same
# bubble afterwards (that path is untouched here), so this only
# removes the bubble popping up on its own during normal scrolling.

class ReadOnlyTextInput(TextInput):

    def long_touch(self, dt):
        pass


# ============================================================
# COURSE CARD
# ============================================================

class CourseCard(BoxLayout):

    def __init__(self, text, **kwargs):

        super().__init__(
            orientation="vertical",
            padding=CARD_PADDING,
            spacing=dp(5),
            size_hint_y=None,
            **kwargs
        )

        # ----------------------------------------------------
        # Background
        # ----------------------------------------------------

        with self.canvas.before:

            Color(*SURFACE_COLOR)

            self.background = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[CARD_RADIUS]
            )

            Color(*BORDER_COLOR)

            self.border = Line(
                rounded_rectangle=(
                    self.x,
                    self.y,
                    self.width,
                    self.height,
                    CARD_RADIUS
                ),
                width=1
            )

        self.bind(
            pos=self.update_background,
            size=self.update_background
        )

        # ----------------------------------------------------
        # Course text (a read-only TextInput, not a Label, so the
        # user can long-press to select and copy course details)
        # ----------------------------------------------------

        self.label = ReadOnlyTextInput(
            text=text,
            readonly=True,
            multiline=True,
            font_size=dp(15),
            foreground_color=TEXT_COLOR,
            background_color=(0, 0, 0, 0),
            cursor_width=0,
            halign="left",
            size_hint_y=None,
            size_hint_x=1,
            padding=(0, 0, 0, 0)
        )

        self.label.bind(
            width=self.update_text_width
        )

        self.label.bind(
            minimum_height=self.update_card_height
        )

        self.add_widget(
            self.label
        )

    # ========================================================
    # UPDATE BACKGROUND
    # ========================================================

    def update_background(
        self,
        *args
    ):

        self.background.pos = self.pos
        self.background.size = self.size

        self.border.rounded_rectangle = (
            self.x,
            self.y,
            self.width,
            self.height,
            CARD_RADIUS
        )

    # ========================================================
    # TEXT WIDTH
    # ========================================================
    # TextInput wraps its own text against its width automatically,
    # unlike Label, so this only needs to nudge it to re-measure.

    def update_text_width(
        self,
        instance,
        width
    ):

        instance.height = instance.minimum_height

    # ========================================================
    # CARD HEIGHT
    # ========================================================

    def update_card_height(
        self,
        instance,
        minimum_height
    ):

        instance.height = minimum_height

        self.height = (
            minimum_height
            + (CARD_PADDING * 2)
        )


# ============================================================
# SOURCE ROW
# ============================================================
# A tappable card for one source: tapping anywhere on the row
# (other than the Copy button) opens the URL in the phone's
# browser; the Copy button copies the URL to the clipboard.
# ButtonBehavior.on_touch_down defers to child widgets first, so
# a touch on the Copy button is consumed there and never reaches
# the row's own on_release.

class SourceRow(ButtonBehavior, BoxLayout):

    def __init__(
        self,
        index,
        title,
        url,
        on_open,
        on_copy,
        **kwargs
    ):

        super().__init__(
            orientation="horizontal",
            padding=CARD_PADDING,
            spacing=dp(10),
            size_hint_y=None,
            **kwargs
        )

        self.url = url

        # ----------------------------------------------------
        # Background
        # ----------------------------------------------------

        with self.canvas.before:

            self._fill_color = Color(*SURFACE_COLOR)

            self.background = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[CARD_RADIUS]
            )

            Color(*BORDER_COLOR)

            self.border = Line(
                rounded_rectangle=(
                    self.x,
                    self.y,
                    self.width,
                    self.height,
                    CARD_RADIUS
                ),
                width=1
            )

        self.bind(
            pos=self.update_background,
            size=self.update_background,
            state=self.update_press_color
        )

        # ----------------------------------------------------
        # Title + URL column
        # ----------------------------------------------------

        text_column = BoxLayout(
            orientation="vertical",
            spacing=dp(3),
            size_hint_x=1,
            size_hint_y=None
        )

        text_column.bind(
            minimum_height=text_column.setter("height")
        )

        title_label = Label(
            text=f"{index}. {title or 'Untitled source'}",
            font_size=dp(14),
            bold=True,
            color=ACCENT_COLOR,
            halign="left",
            valign="top",
            size_hint_y=None
        )

        title_label.bind(
            width=lambda instance, width:
            setattr(instance, "text_size", (width, None))
        )

        title_label.bind(
            texture_size=lambda instance, size:
            setattr(instance, "height", size[1])
        )

        url_label = Label(
            text=url,
            font_size=dp(12),
            color=MUTED_TEXT_COLOR,
            halign="left",
            valign="top",
            shorten=True,
            shorten_from="right",
            size_hint_y=None
        )

        url_label.bind(
            width=lambda instance, width:
            setattr(instance, "text_size", (width, None))
        )

        url_label.bind(
            texture_size=lambda instance, size:
            setattr(instance, "height", size[1])
        )

        text_column.add_widget(title_label)
        text_column.add_widget(url_label)

        text_column.bind(
            height=self.update_row_height
        )

        # ----------------------------------------------------
        # Copy button
        # ----------------------------------------------------

        copy_button = RoundedButton(
            fill_color=BORDER_COLOR,
            text="Copy",
            font_size=dp(12),
            size_hint=(None, None),
            width=dp(64),
            height=dp(34)
        )

        copy_button.bind(
            on_press=lambda instance: on_copy(url)
        )

        self.add_widget(text_column)
        self.add_widget(copy_button)

        self.bind(
            on_release=lambda instance: on_open(url)
        )

        self.update_row_height(
            text_column,
            text_column.height
        )

    # ========================================================
    # UPDATE BACKGROUND
    # ========================================================

    def update_background(
        self,
        *args
    ):

        self.background.pos = self.pos
        self.background.size = self.size

        self.border.rounded_rectangle = (
            self.x,
            self.y,
            self.width,
            self.height,
            CARD_RADIUS
        )

    # ========================================================
    # PRESS FEEDBACK
    # ========================================================

    def update_press_color(
        self,
        *args
    ):

        if self.state == "down":

            self._fill_color.rgba = tuple(
                min(1, channel + 0.04)
                for channel in SURFACE_COLOR[:3]
            ) + (SURFACE_COLOR[3],)

        else:

            self._fill_color.rgba = SURFACE_COLOR

    # ========================================================
    # ROW HEIGHT
    # ========================================================

    def update_row_height(
        self,
        instance,
        height
    ):

        self.height = max(
            height + (CARD_PADDING * 2),
            dp(34) + (CARD_PADDING * 2)
        )


# ============================================================
# KERALA IT HUB APP
# ============================================================

class KeralaITHubApp(App):

    # ========================================================
    # BUILD
    # ========================================================

    def build(self):

        # ----------------------------------------------------
        # Load persisted settings (backend URL / API key)
        # ----------------------------------------------------

        settings = load_settings()

        self.api_base_url = settings["api_base_url"]
        self.api_key = settings["api_key"]

        self._search_clock_event = None

        # ----------------------------------------------------
        # Main layout
        # ----------------------------------------------------

        main_layout = BoxLayout(
            orientation="vertical",
            padding=dp(20),
            spacing=dp(10)
        )

        # ====================================================
        # TOP BAR (TITLE + SETTINGS)
        # ====================================================

        top_bar = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(55)
        )

        title = Label(
            text="KERALA IT HUB",
            font_size=dp(30),
            bold=True,
            color=ACCENT_COLOR,
            halign="left",
            valign="middle"
        )

        title.bind(
            size=lambda instance, value:
            setattr(
                instance,
                "text_size",
                (value[0], None)
            )
        )

        settings_button = RoundedButton(
            fill_color=SURFACE_COLOR,
            text="Settings",
            font_size=dp(13),
            size_hint_x=None,
            width=dp(90)
        )

        settings_button.bind(
            on_press=self.open_settings
        )

        top_bar.add_widget(title)
        top_bar.add_widget(settings_button)

        # ====================================================
        # SUBTITLE
        # ====================================================

        subtitle = Label(
            text=(
                "AI-Powered IT Course & "
                "Institute Navigator"
            ),
            font_size=dp(16),
            color=MUTED_TEXT_COLOR,
            size_hint_y=None,
            height=dp(35)
        )

        # ====================================================
        # QUESTION LABEL
        # ====================================================

        question_label = Label(
            text="Ask your question:",
            font_size=dp(17),
            halign="left",
            valign="middle",
            size_hint_y=None,
            height=dp(30)
        )

        question_label.bind(
            size=lambda instance, value:
            setattr(
                instance,
                "text_size",
                (value[0], None)
            )
        )

        # ====================================================
        # QUESTION INPUT
        # ====================================================

        self.question_input = TextInput(
            hint_text=(
                "Example: Which Python courses "
                "are available in Kerala?"
            ),
            multiline=True,
            font_size=dp(16),
            background_color=SURFACE_COLOR,
            foreground_color=TEXT_COLOR,
            hint_text_color=MUTED_TEXT_COLOR,
            cursor_color=ACCENT_COLOR,
            size_hint_y=None,
            height=dp(75),
            padding=dp(10)
        )

        # ====================================================
        # ASK BUTTON
        # ====================================================

        self.ask_button = RoundedButton(
            fill_color=ACCENT_COLOR,
            text="ASK",
            font_size=dp(18),
            bold=True,
            size_hint_y=None,
            height=dp(48)
        )

        self.ask_button.bind(
            on_press=self.ask_question
        )

        # ====================================================
        # STATUS
        # ====================================================

        self.status_label = Label(
            text="",
            font_size=dp(14),
            color=MUTED_TEXT_COLOR,
            halign="center",
            valign="middle",
            size_hint_y=None,
            height=dp(28)
        )

        # ====================================================
        # ANSWER TITLE
        # ====================================================

        answer_title = Label(
            text="Answer",
            font_size=dp(20),
            bold=True,
            color=TEXT_COLOR,
            halign="left",
            valign="middle",
            size_hint_y=None,
            height=dp(32)
        )

        answer_title.bind(
            size=lambda instance, value:
            setattr(
                instance,
                "text_size",
                (value[0], None)
            )
        )

        # ====================================================
        # ANSWER SCROLL
        # ====================================================

        self.answer_scroll = ScrollView(
            size_hint_y=1,
            do_scroll_x=False,
            do_scroll_y=True,
            bar_width=dp(7)
        )

        # ====================================================
        # ANSWER LAYOUT
        # ====================================================

        self.answer_layout = BoxLayout(
            orientation="vertical",
            spacing=dp(12),
            padding=(dp(5), dp(5)),
            size_hint_y=None
        )

        self.answer_layout.bind(
            minimum_height=
            self.answer_layout.setter(
                "height"
            )
        )

        self.answer_scroll.add_widget(
            self.answer_layout
        )

        # ====================================================
        # ADD WIDGETS
        # ====================================================
        # Answer and Sources now share this single scrollable
        # feed (display_sources appends into answer_layout after
        # display_answer), instead of Sources being a separate,
        # awkwardly small fixed-height scroll box of its own.

        main_layout.add_widget(top_bar)
        main_layout.add_widget(subtitle)
        main_layout.add_widget(question_label)
        main_layout.add_widget(self.question_input)
        main_layout.add_widget(self.ask_button)
        main_layout.add_widget(self.status_label)
        main_layout.add_widget(answer_title)
        main_layout.add_widget(self.answer_scroll)

        return main_layout

    # ========================================================
    # OPEN SETTINGS
    # ========================================================

    def open_settings(
        self,
        instance
    ):

        form = BoxLayout(
            orientation="vertical",
            spacing=dp(10),
            padding=dp(15)
        )

        form.add_widget(
            Label(
                text="Backend server URL",
                font_size=dp(14),
                color=MUTED_TEXT_COLOR,
                size_hint_y=None,
                height=dp(22),
                halign="left"
            )
        )

        url_input = TextInput(
            text=self.api_base_url,
            multiline=False,
            font_size=dp(14),
            background_color=SURFACE_COLOR,
            foreground_color=TEXT_COLOR,
            cursor_color=ACCENT_COLOR,
            size_hint_y=None,
            height=dp(42)
        )

        form.add_widget(url_input)

        form.add_widget(
            Label(
                text="API key (leave blank if none)",
                font_size=dp(14),
                color=MUTED_TEXT_COLOR,
                size_hint_y=None,
                height=dp(22),
                halign="left"
            )
        )

        key_input = TextInput(
            text=self.api_key,
            multiline=False,
            password=True,
            font_size=dp(14),
            background_color=SURFACE_COLOR,
            foreground_color=TEXT_COLOR,
            cursor_color=ACCENT_COLOR,
            size_hint_y=None,
            height=dp(42)
        )

        form.add_widget(key_input)

        button_row = BoxLayout(
            orientation="horizontal",
            spacing=dp(10),
            size_hint_y=None,
            height=dp(45)
        )

        save_button = RoundedButton(
            fill_color=ACCENT_COLOR,
            text="Save"
        )

        cancel_button = RoundedButton(
            fill_color=SURFACE_COLOR,
            text="Cancel"
        )

        button_row.add_widget(cancel_button)
        button_row.add_widget(save_button)

        form.add_widget(button_row)

        popup = Popup(
            title="Settings",
            title_color=TEXT_COLOR,
            separator_color=ACCENT_COLOR,
            background_color=SURFACE_COLOR,
            content=form,
            size_hint=(0.9, 0.5)
        )

        def on_save(_instance):

            new_url = url_input.text.strip().rstrip("/")
            new_key = key_input.text.strip()

            if not new_url:
                return

            self.api_base_url = new_url
            self.api_key = new_key

            save_settings(new_url, new_key)

            popup.dismiss()

        save_button.bind(on_press=on_save)
        cancel_button.bind(on_press=popup.dismiss)

        popup.open()

    # ========================================================
    # CLEAN MARKDOWN
    # ========================================================

    def clean_markdown(
        self,
        text
    ):

        if not text:
            return ""

        text = text.replace(
            "**",
            ""
        )

        text = text.replace(
            "__",
            ""
        )

        text = text.replace(
            "<br>",
            "\n"
        )

        text = text.replace(
            "<br/>",
            "\n"
        )

        text = text.replace(
            "<br />",
            "\n"
        )

        return text.strip()

    # ========================================================
    # NORMAL TEXT (read-only + copyable)
    # ========================================================
    # A read-only TextInput rather than a Label, so the user can
    # long-press to select and copy the answer / error text.

    def create_text_label(
        self,
        text,
        font_size=15
    ):

        text_input = ReadOnlyTextInput(
            text=text,
            readonly=True,
            multiline=True,
            font_size=dp(font_size),
            foreground_color=TEXT_COLOR,
            background_color=(0, 0, 0, 0),
            cursor_width=0,
            halign="left",
            size_hint_y=None,
            size_hint_x=1,
            padding=(dp(10), dp(8), dp(10), dp(8))
        )

        text_input.bind(
            minimum_height=text_input.setter("height")
        )

        return text_input

    # ========================================================
    # SECTION HEADING (not copyable, just a label)
    # ========================================================

    def create_heading_label(
        self,
        text,
        font_size=18
    ):

        label = Label(
            text=text,
            font_size=dp(font_size),
            bold=True,
            color=TEXT_COLOR,
            halign="left",
            valign="middle",
            size_hint_y=None,
            height=dp(font_size) + dp(20)
        )

        label.bind(
            size=lambda instance, value:
            setattr(
                instance,
                "text_size",
                (value[0], None)
            )
        )

        return label

    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    def display_answer(
        self,
        answer
    ):

        self.answer_layout.clear_widgets()

        if not answer:

            self.answer_layout.add_widget(
                self.create_text_label(
                    "No answer available."
                )
            )

            return

        answer = self.clean_markdown(
            answer
        )

        lines = answer.splitlines()

        table_rows = []
        normal_lines = []

        # ----------------------------------------------------
        # Process answer
        # ----------------------------------------------------

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # ------------------------------------------------
            # Detect markdown table
            # ------------------------------------------------

            if "|" in line:

                parts = [
                    item.strip()
                    for item in
                    line.strip("|").split("|")
                ]

                parts = [
                    item
                    for item in parts
                    if item
                ]

                # Ignore separator
                if all(
                    re.fullmatch(
                        r"[-:]+",
                        item.replace(" ", "")
                    )
                    for item in parts
                ):
                    continue

                # A source citation ("1. Title - https://...") can
                # end up containing a "-" that this loop wouldn't
                # catch, but it always contains a URL -- institute
                # and course names don't. Guard against the model
                # occasionally slipping a source line into table
                # format instead of the requested plain list.
                looks_like_source_citation = any(
                    "http://" in item or "https://" in item
                    for item in parts
                )

                if len(parts) >= 2 and not looks_like_source_citation:

                    table_rows.append(
                        parts
                    )

                    continue

            # ------------------------------------------------
            # Normal text
            # ------------------------------------------------

            normal_lines.append(
                line
            )

        # ====================================================
        # DISPLAY INTRODUCTION
        # ====================================================

        if normal_lines:

            intro_text = []

            for line in normal_lines:

                # Avoid displaying source section
                if line.lower() == "sources":
                    continue

                intro_text.append(line)

            if intro_text:

                self.answer_layout.add_widget(
                    self.create_text_label(
                        "\n\n".join(intro_text),
                        font_size=15
                    )
                )

        # ====================================================
        # DISPLAY COURSE CARDS
        # ====================================================

        if table_rows:

            # ------------------------------------------------
            # Section heading
            # ------------------------------------------------

            heading = self.create_heading_label(
                "Courses found:",
                font_size=18
            )

            self.answer_layout.add_widget(
                heading
            )

            header = table_rows[0]

            # ------------------------------------------------
            # Each row becomes a card
            # ------------------------------------------------

            for row in table_rows[1:]:

                if not row:
                    continue

                card_lines = []

                for index, value in enumerate(row):

                    if index >= len(header):
                        continue

                    field = header[index]

                    if not value:
                        continue

                    # ------------------------------------------------
                    # Make field name cleaner
                    # ------------------------------------------------

                    field = field.strip()

                    card_lines.append(
                        f"{field}: {value}"
                    )

                if card_lines:

                    card_text = "\n".join(
                        card_lines
                    )

                    card = CourseCard(
                        card_text
                    )

                    self.answer_layout.add_widget(
                        card
                    )

        # ====================================================
        # FALLBACK
        # ====================================================

        if not table_rows and normal_lines:

            # Already displayed above
            pass

    # ========================================================
    # MUTED LABEL (a short static message, not user content --
    # no need for the copyable-TextInput treatment)
    # ========================================================

    def create_muted_label(
        self,
        text
    ):

        label = Label(
            text=text,
            font_size=dp(13),
            color=MUTED_TEXT_COLOR,
            halign="left",
            valign="top",
            size_hint_y=None,
            padding=(0, dp(4))
        )

        label.bind(
            width=lambda instance, width:
            setattr(instance, "text_size", (width, None))
        )

        label.bind(
            texture_size=lambda instance, size:
            setattr(instance, "height", size[1])
        )

        return label

    # ========================================================
    # DISPLAY SOURCES
    # ========================================================
    # Appended into answer_layout after display_answer, so
    # Answer and Sources share one continuous, aligned scroll
    # feed instead of two separately-scrolled sections.

    def display_sources(
        self,
        sources
    ):

        self.answer_layout.add_widget(
            self.create_heading_label(
                "Sources",
                font_size=18
            )
        )

        if not sources:

            self.answer_layout.add_widget(
                self.create_muted_label(
                    "No sources available."
                )
            )

            return

        for index, source in enumerate(
            sources,
            start=1
        ):

            url = source.get(
                "url",
                ""
            )

            if not url:
                continue

            title = source.get(
                "title",
                "Untitled source"
            )

            row = SourceRow(
                index=index,
                title=title,
                url=url,
                on_open=self.open_url,
                on_copy=self.copy_source_url
            )

            self.answer_layout.add_widget(
                row
            )

    # ========================================================
    # OPEN URL
    # ========================================================
    # webbrowser.open() has no way to launch a browser on
    # Android, so route through an explicit Intent there.

    def open_url(
        self,
        url
    ):

        if not url:
            return

        try:

            if platform == "android":

                from jnius import autoclass

                Intent = autoclass("android.content.Intent")
                Uri = autoclass("android.net.Uri")
                PythonActivity = autoclass(
                    "org.kivy.android.PythonActivity"
                )

                intent = Intent(
                    Intent.ACTION_VIEW,
                    Uri.parse(url)
                )

                intent.addFlags(
                    Intent.FLAG_ACTIVITY_NEW_TASK
                )

                PythonActivity.mActivity.startActivity(
                    intent
                )

            else:

                import webbrowser

                webbrowser.open(url)

        except Exception:

            self.status_label.text = (
                "Could not open the link."
            )

    # ========================================================
    # COPY SOURCE URL
    # ========================================================

    def copy_source_url(
        self,
        url
    ):

        Clipboard.copy(url)

        self.status_label.text = (
            "Link copied to clipboard."
        )

    # ========================================================
    # ASK QUESTION
    # ========================================================

    def ask_question(
        self,
        instance
    ):

        question = (
            self.question_input
            .text
            .strip()
        )

        if not question:

            self.status_label.text = (
                "Please enter a question."
            )

            return

        if (
            not self.api_base_url
            or "YOUR-BACKEND-URL" in self.api_base_url
        ):

            self.status_label.text = (
                "Please set the backend server URL "
                "in Settings first."
            )

            return

        # ----------------------------------------------------
        # Disable button
        # ----------------------------------------------------

        self.ask_button.disabled = True

        self.ask_button.text = (
            "SEARCHING..."
        )

        # ----------------------------------------------------
        # Loading
        # ----------------------------------------------------

        self.answer_layout.clear_widgets()

        self.answer_layout.add_widget(
            self.create_text_label(
                (
                    "Searching the web and retrieving "
                    "relevant course information..."
                ),
                font_size=15
            )
        )

        # ----------------------------------------------------
        # Elapsed-time ticker, so a slow response (the backend does
        # a handful of live web searches plus an LLM call, and can
        # take a couple of minutes -- longer still if the free-tier
        # Render service was idle and needs ~30-60s just to wake up)
        # still reads as "working", not "stuck".
        # ----------------------------------------------------

        self._search_seconds = 0

        self.status_label.text = "Searching... (0s)"

        self._search_clock_event = Clock.schedule_interval(
            self.tick_search_status,
            1
        )

        # ----------------------------------------------------
        # Thread
        # ----------------------------------------------------

        thread = threading.Thread(
            target=self.send_question_to_api,
            args=(question,),
            daemon=True
        )

        thread.start()

    # ========================================================
    # TICK SEARCH STATUS
    # ========================================================

    def tick_search_status(
        self,
        dt
    ):

        self._search_seconds += 1

        message = f"Searching... ({self._search_seconds}s)"

        if self._search_seconds == 20:
            message += " -- still working, this can take a minute or two"
        elif self._search_seconds == 45:
            message += (
                " -- if the server was idle, it can take up to a "
                "minute just to wake up"
            )
        elif self._search_seconds == 100:
            message += (
                " -- comparison questions check more sources, "
                "so this takes longer than a simple lookup"
            )

        self.status_label.text = message

    # ========================================================
    # API REQUEST
    # ========================================================

    def send_question_to_api(
        self,
        question
    ):

        try:

            headers = {}

            if self.api_key:
                headers["X-API-Key"] = self.api_key

            response = requests.post(
                f"{self.api_base_url}/ask",
                json={
                    "question": question
                },
                headers=headers,
                # Comparison-style questions now search up to 10 pages
                # server-side (see backend/main.py's wants_comparison),
                # which combined with a cold Render free-tier wake-up
                # can take longer than the previous 180s allowed.
                timeout=280
            )

            # ------------------------------------------------
            # HTTP error
            # ------------------------------------------------

            if response.status_code != 200:

                self.update_ui(
                    "error",
                    (
                        "FastAPI returned "
                        f"status code "
                        f"{response.status_code}."
                    ),
                    []
                )

                return

            # ------------------------------------------------
            # JSON
            # ------------------------------------------------

            data = response.json()

            # ------------------------------------------------
            # API error
            # ------------------------------------------------

            if data.get("status") != "success":

                self.update_ui(
                    "error",
                    data.get(
                        "message",
                        "Unknown API error."
                    ),
                    []
                )

                return

            # ------------------------------------------------
            # Answer
            # ------------------------------------------------

            answer = data.get(
                "answer",
                "No answer returned."
            )

            # ------------------------------------------------
            # Sources
            # ------------------------------------------------

            sources = data.get(
                "sources",
                []
            )

            # ------------------------------------------------
            # Update UI
            # ------------------------------------------------

            self.update_ui(
                "success",
                answer,
                sources
            )

        # ----------------------------------------------------
        # Connection error
        # ----------------------------------------------------

        except requests.exceptions.ConnectionError:

            self.update_ui(
                "error",
                (
                    "Could not connect to FastAPI.\n\n"
                    "Please make sure the server is running."
                ),
                []
            )

        # ----------------------------------------------------
        # Timeout
        # ----------------------------------------------------

        except requests.exceptions.Timeout:

            self.update_ui(
                "error",
                (
                    "The request took too long.\n\n"
                    "Please try again."
                ),
                []
            )

        # ----------------------------------------------------
        # Other error
        # ----------------------------------------------------

        except Exception as e:

            self.update_ui(
                "error",
                f"Error: {str(e)}",
                []
            )

    # ========================================================
    # UPDATE UI
    # ========================================================

    def update_ui(
        self,
        status,
        answer,
        sources
    ):

        Clock.schedule_once(
            lambda dt:
            self.finish_ui_update(
                status,
                answer,
                sources
            )
        )

    # ========================================================
    # FINISH UI
    # ========================================================

    def finish_ui_update(
        self,
        status,
        answer,
        sources
    ):

        # ----------------------------------------------------
        # Stop the elapsed-time ticker started in ask_question
        # ----------------------------------------------------

        if self._search_clock_event is not None:
            self._search_clock_event.cancel()
            self._search_clock_event = None

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if status == "success":

            self.status_label.text = (
                "Answer generated successfully."
            )

        else:

            self.status_label.text = (
                "Request failed."
            )

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        if status == "success":

            self.display_answer(
                answer
            )

            self.display_sources(
                sources
            )

        else:

            self.answer_layout.clear_widgets()

            self.answer_layout.add_widget(
                self.create_text_label(
                    answer,
                    font_size=15
                )
            )

        # ----------------------------------------------------
        # Reset scroll
        # ----------------------------------------------------

        Clock.schedule_once(
            self.reset_scroll,
            0.5
        )

        # ----------------------------------------------------
        # Enable button
        # ----------------------------------------------------

        self.ask_button.disabled = False

        self.ask_button.text = "ASK"

    # ========================================================
    # RESET SCROLL
    # ========================================================

    def reset_scroll(
        self,
        dt
    ):

        self.answer_scroll.scroll_y = 1


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":

    KeralaITHubApp().run()