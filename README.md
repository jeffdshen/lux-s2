# lux-s2

## Common commands

### copy files to bot folder
```
mkdir bots/v0.0.X
cp -R anim lux agent.py main.py bots/v0.0.X
```

### make submission
```
(cd bots/v0.0.X && tar -czvf ../../submission.tar.gz *)
```
The parentheses are important if you want to end up back in the same directory.