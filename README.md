# RaycastMaze ( Drafting ... )
RaycastMaze for current PyGame and PyDraw versions 2.1.0

### Game problem and re-solution ###
Decide from game input to output ( solution ), continue of arrays with dimension ( 512, 512, 3 ) streams as running of both side of wall object and fencing random when player move with PyGame constant ```actions = { "none_1": K_h, "left_1": K_a, "down_1": K_s, "right1": K_d, "up___1": K_w }```

The problem is find a red block from both side of walls, input is array of screens and action. By summing up multiple records of screen and action you can predict the result from one record input to find a red block as the game purpose. ``` screen + action = YES|NO ``` Apply this problem, you can identify of inputs to result by adding some value as white colour to red colour ( R, G B ) arrays.

🧸💬 Scores or not scores system depending on catagory type see this input ``` dataset = tf.constant([ DATA, LABEL ]) ``` when DATA is ``` tf.constant([ dim_1, dim_2, dim_3 ]) ``` mean single data or you can create stack input with ``` DATA = tf.experimental.numpy.vstack([DATA, DATA_row]) ```

🐑💬 Masking is helpful ``` masking_layer = tf.keras.layers.Masking(mask_value=50, input_shape=(n_horizon, n_features)) ```

👧💬 Sufficient resources do not waste time convert input and output shape use stack and masking, the data is remain in shape you may try to use tf.reshape to shape A and convert back with the same method it will not have the full size picture.

🐐💬 You talked about that we found another sample when usin USB compatibility, camera and voice stream make sure you use it correct data transfer rates per versions.
( The problem of Repoo and Integrated camera and Google compatibility does not cause by Chrome or IE but the USB transfer rates, test it with mobile networks before use )

| Problem types | Description | Learning speed AI, unsupervised |
| --- | --- | --- |
| Identify significant object | Sample arcades games with CONV layers input | Fast, Required data preparation |
| Catagorize | Picture or camera object detection | Medium, Does not required data preparation but dimension |
| Maze with rules | Sample arcades games | Very fast, pattern finding |
| Maze no rules | Sample arcardes games screen input | Medium, Does not required data preparation but dimension  |

### Correct function map with PyDraw Input definition ###

| Functions | Description |
| --- | --- |
| line(surface, color, start_pos, end_pos) -> Rect | Draws a straight line on the given surface. There are no endcaps. For thick lines the ends are squared off. |
| line(surface, color, start_pos, end_pos, width=1) -> Rect | Draws a straight line on the given surface. There are no endcaps. For thick lines the ends are squared off. |
| start_pos (tuple(int or float, int or float) or list(int or float, int or float) or Vector2(int or float, int or float)) | start position of the line, (x, y) |
| end_pos (tuple(int or float, int or float) or list(int or float, int or float) or Vector2(int or float, int or float)) | end position of the line, (x, y) |


### raycastmaze.py ###

```
def step(self, dt):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (92, 92, 92),
                         (0, self.height / 2, self.width, self.height))

        if not self.is_game_over:
            self.score += self.rewards["tick"]
            self._handle_player_events(dt)

            c, t, b, col = self.draw()

            for i in range(len(c)):
                color = (col[i][0], col[i][1], col[i][2])
                # p0 = (c[i], t[i]) # <-- remove the input based on the input spec in Ref[0].
                # p1 = (c[i], b[i]) # <-- remove the input based on the input spec in Ref[0].

                p0 = (c[i], int(t[i])) # <-- add the input based on the input spec in Ref[0].
                p1 = (c[i], int(b[i])) # <-- add the input based on the input spec in Ref[0].
                pygame.draw.line(self.screen, color, p0, p1, self.resolution)

            dist = np.sqrt(np.sum((self.pos[0] - (self.obj_loc[0] + 0.5))**2.0))
            # Close to target object and in sight
            if dist < 1.1 and self.angle_to_obj_rad() < 0.8:
                self.score += self.rewards["win"]
                self.is_game_over = True
```

### RaycastMaze.py ###

```
import ple
from ple import PLE
# from ple.games.raycast import RayCastPlayer # Player class
from ple.games.raycastmaze import RaycastMaze # Game class
from pygame.constants import K_w, K_a, K_d, K_s, K_h # KeyPad constants
```

### Model ###

```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Initialize
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
input_shape = (42, 42, 3)

model = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=input_shape),
	
	tf.keras.layers.Reshape((1, 42 * 42 * 3)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, return_state=False)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))

])
		
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(192))
model.add(tf.keras.layers.Dense(5))
model.summary()
```

### Files and Directory ###
| File name | Description |
| --- | --- |
| 01.png | Screen shot |
| ezgif.com-video-to-gif.gif | GIF image |
| sample.py | Sample Codes |
| README.md | Readme file |


### References ###
Ref[0]: https://www.pygame.org/docs/ref/draw.html#pygame.draw.line

### Results ###
Some sample output from target codes running from our Notepad.

#### Start from initial game and test it  ####
Solved first problem, game does not compatible with current PyGame versions by PyDraw input definition.

![Sample](https://github.com/jkaewprateep/RaycastMaze/blob/main/01.png "Sample")

#### Start running ####
AI understand the problem instant and find the solution by find the most contrast from its input.

![Sample](https://github.com/jkaewprateep/RaycastMaze/blob/main/ezgif.com-video-to-gif.gif "Sample")


