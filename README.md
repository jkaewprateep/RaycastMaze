# RaycastMaze ( Drafting ... )
RaycastMaze for current PyGame and PyDraw versions

```
for i in range(len(c)):
    color = (col[i][0], col[i][1], col[i][2])
    p0 = (c[i], t[i]) # <-- remove the input based on the input spec in Ref[0].
    p1 = (c[i], b[i]) # <-- remove the input based on the input spec in Ref[0].

    p0 = (c[i], int(t[i])) # <-- add the input based on the input spec in Ref[0].
    p1 = (c[i], int(b[i])) # <-- add the input based on the input spec in Ref[0].
    pygame.draw.line(self.screen, color, p0, p1, self.resolution)
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
