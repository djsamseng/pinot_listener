# Pinot Listener

```bash
$ python3 main.py
# OR
$ python3 main_ai.py
```

## Repeater.py
```bash
$ python3 repeater.py
```
### Commands
- `r h`: start recording to h.json
- `s`: stop recording and save file
- `p h`: play h.json

### Interactive
```bash
$ python3
```
```python3
import repeater as r
q = r.data_manager_api()
data = r.gather_until_exit(q.audio_record_queue)
r.play_audio_data(q.audio_play_queue, data)
r.save_audio_data("h", data)
c = r.read_concepts()
r.play_audio_data(q.audio_play_queue, c.h)
```

