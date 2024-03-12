import re


class BeatStringProcessor:
    def __init__(self, 
        beat: bool,
        downbeat: bool
    ):
        self.beat = beat
        self.downbeat = downbeat

    def events_to_strings(self, events):

        strings = ["<sos>"]

        for e in events:

            if e["name"] == "beat":
                if self.beat:
                    strings = self.append_name(strings, e["name"])
                    strings = self.append_time(strings, e["time"])
                    strings = self.append_beat_index(strings, e["index"])

            elif e["name"] == "downbeat":
                if self.downbeat:
                    strings = self.append_name(strings, e["name"])
                    strings = self.append_time(strings, e["time"])

            else:
                raise NotImplementedError

        strings.append("<eos>")
        
        return strings

    def strings_to_events(self, strings):

        events = []

        i = 0

        while i < len(strings):

            if "=" in strings[i]:
                key = re.search('(.*)=', strings[i]).group(1)
                value = re.search('=(.*)', strings[i]).group(1)
                value = self.format_value(key, value)

                if key == "name":
                    event = {key: value}
                    event, shift = self.look_forward(strings, i, event)
                    events.append(event)
                    i += shift
                    continue

            i += 1

        return events

    def look_forward(self, strings, i, event):

        for j in range(1, 10):
            if strings[i + j] == "<eos>":
                return event, j

            next_key = re.search('(.*)=', strings[i + j]).group(1)
            next_value = re.search('=(.*)', strings[i + j]).group(1)
            next_value = self.format_value(next_key, next_value)

            if next_key == "name":
                return event, j
            else:
                event[next_key] = next_value


    def append_name(self, strings, name):
        
        strings.append("name={}".format(name))

        return strings

    def append_time(self, strings, time):
        
        strings.append("time={}".format(time))

        return strings

    def append_beat_index(self, strings, beat_index):
        
        strings.append("beat_index={}".format(beat_index))

        return strings

    def format_value(self, key, value):
        if key in ["time"]:
            return float(value)

        elif key in ["beat_index"]:
            return int(value)

        else:
            return value