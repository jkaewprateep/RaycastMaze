# RaycastMaze ( Drafting ... )
RaycastMaze for current PyGame and PyDraw versions

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

```
import ple
from ple import PLE
# from ple.games.raycast import RayCastPlayer # Player class
from ple.games.raycastmaze import RaycastMaze # Game class
from pygame.constants import K_w, K_a, K_d, K_s, K_h # KeyPad constants
```

### Files and Directory ###
| File name | Description |
| --- | --- |
| 01.png | Screen shot |
| sample.py | Sample Codes |
| README.md | Readme file |

### References ###
Ref[0]: https://www.pygame.org/docs/ref/draw.html#pygame.draw.line

![Sample](https://github.com/jkaewprateep/RaycastMaze/blob/main/01.png "Sample")
