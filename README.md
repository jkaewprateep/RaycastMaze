# RaycastMaze ( Drafting ... )
RaycastMaze for current PyGame and PyDraw versions

```
for i in range(len(c)):
    color = (col[i][0], col[i][1], col[i][2])
    p0 = (c[i], t[i]) # <-- remove the input based on the input spec. 
    p1 = (c[i], b[i])

    p0 = (c[i], int(t[i]))
    p1 = (c[i], int(b[i]))
    pygame.draw.line(self.screen, color, p0, p1, self.resolution)
```

Ref[0]: https://www.pygame.org/docs/ref/draw.html#pygame.draw.line

![Sample](https://github.com/jkaewprateep/RaycastMaze/blob/main/01.png "Sample")
