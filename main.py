from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import NumericProperty

from PIL import Image, ImageDraw, ImageFilter

from math import hypot
from kivy.lang import Builder

from kivy.graphics import Color, Line


FILENAME = 'handwriting.jpg'


class Painter(Widget):
    pencil_size = NumericProperty(35)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.canvas_size = (int(self.width), int(self.height))
        self.image = Image.new("RGB", self.canvas_size, "white")
        self.draw = ImageDraw.Draw(self.image)
        self.drawing = False
        self.last_pos = None

    def on_size(self, *args):
        # Create a new image with the new size
        new_size = (int(self.width), int(self.height))
        if new_size == self.image.size:
            return
            
        # Create new blank image
        new_image = Image.new("RGB", new_size, "white")
        
        # Paste the old image scaled to new size
        if self.image.size != (0, 0):
            self.image = self.image.resize(new_size, Image.Resampling.LANCZOS)
            new_image.paste(self.image, (0, 0))
            
        self.image = new_image
        self.draw = ImageDraw.Draw(self.image)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.drawing = True
            self.last_pos = touch.pos

    def on_touch_move(self, touch):
        if self.drawing and self.last_pos:
            with self.canvas:
                Color(0, 0, 0)
                Line(points=[*self.last_pos, *touch.pos], width=self.pencil_size)

            # Draw filled circles along the path for PIL image
            self._draw_brush_line(self.last_pos, touch.pos)
            self.last_pos = touch.pos

    def _draw_brush_line(self, start, end):
        x0, y0 = start
        x1, y1 = end
        y0 = self.height - y0  # Convert Kivy y-coordinate to PIL y-coordinate
        y1 = self.height - y1
        
        distance = int(hypot(x1 - x0, y1 - y0))
        if distance == 0:
            distance = 1  # Prevent division by zero

        for i in range(distance):
            t = i / distance
            x = int(x0 + (x1 - x0) * t)
            y = int(y0 + (y1 - y0) * t)
            r = int(self.pencil_size // 2)
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill='black')

    def on_touch_up(self, touch):
        self.drawing = False
        self.last_pos = None

    def clear_canvas(self):
        self.image = Image.new("RGB", (int(self.width), int(self.height)), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.clear()

    def save_image(self):
        smoothed = self.image.filter(ImageFilter.SMOOTH_MORE)
        smoothed.save(FILENAME)
        popup = Popup(title="Saved",
                      content=Label(text=f"Saved as {FILENAME}"),
                      size_hint=(0.4, 0.3))
        popup.open()


class PainterApp(App):
    def build(self):
        return Builder.load_file("painter_main.kv")


if __name__ == '__main__':
    PainterApp().run()