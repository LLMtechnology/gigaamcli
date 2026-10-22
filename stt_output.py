import bisect
import heapq

import config


def _covered_seconds(intervals):
    if not intervals:
        return 0.0
    intervals.sort()
    start, end = intervals[0]
    duration = 0.0
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            duration += end - start
            start, end = next_start, next_end
    return duration + end - start


def assign_speakers(words, turns):
    ordered_turns = sorted(turns, key=lambda turn: (turn["start"], turn["end"], turn["speaker"]))
    end_order = sorted(ordered_turns, key=lambda turn: (turn["end"], turn["start"]))
    start_times = [turn["start"] for turn in ordered_turns]
    end_times = [turn["end"] for turn in end_order]
    active = {}
    expiry = []
    cursor = 0
    labels = {}
    result = []
    for word in sorted(words, key=lambda item: item["start"]):
        start, end = float(word["start"]), float(word["end"])
        while cursor < len(ordered_turns) and ordered_turns[cursor]["start"] <= end:
            turn = ordered_turns[cursor]
            active[cursor] = turn
            heapq.heappush(expiry, (turn["end"], cursor))
            cursor += 1
        while expiry and expiry[0][0] <= start:
            _, expired = heapq.heappop(expiry)
            active.pop(expired, None)
        coverage = {}
        for turn in active.values():
            left, right = max(start, turn["start"]), min(end, turn["end"])
            if right > left:
                coverage.setdefault(turn["speaker"], []).append((left, right))
        durations = {speaker: _covered_seconds(intervals) for speaker, intervals in coverage.items()}
        speaker = max(durations, key=durations.get) if durations else None
        if speaker is None and end == start:
            containing = [turn for turn in active.values() if turn["start"] <= start < turn["end"]]
            if containing:
                speaker = containing[0]["speaker"]
        if speaker is None and ordered_turns:
            candidates = []
            previous = bisect.bisect_right(end_times, start) - 1
            following = bisect.bisect_left(start_times, end)
            if previous >= 0:
                candidates.append((start - end_times[previous], end_order[previous]))
            if following < len(ordered_turns):
                candidates.append((start_times[following] - end, ordered_turns[following]))
            if candidates:
                distance, nearest = min(candidates, key=lambda candidate: candidate[0])
                if distance <= 1.0:
                    speaker = nearest["speaker"]
        if speaker is not None:
            labels.setdefault(speaker, len(labels) + 1)
            speaker = labels[speaker]
        overlap = sum(duration > (end - start) / 2 for duration in durations.values()) > 1
        result.append({**word, "speaker": speaker, "overlap": overlap})
    return result


def _append_text(text, token):
    token = token.strip()
    if not text or not token:
        return text + token
    if token[0] in ".,;:!?…%)]}»”’" or text[-1] in "([{«“‘":
        return text + token
    if token[0] == "'" or text[-1] in "'’":
        return text + token
    return text + " " + token


def segments_from_words(words, diarize=True):
    segments = []
    current = None
    for word in sorted(words, key=lambda item: item["start"]):
        if not word["text"].strip():
            continue
        start, end = float(word["start"]), float(word["end"])
        speaker = word.get("speaker") if diarize else None
        new_segment = current is None
        if current is not None:
            new_segment = any((
                speaker != current["speaker"],
                start - current["end"] >= config.PHRASE_PAUSE_SECONDS,
                end - current["start"] >= config.MAX_PHRASE_SECONDS,
            ))
        if new_segment:
            current = {"start": start, "end": end, "speaker": speaker, "text": "", "overlap": False}
            segments.append(current)
        current["end"] = max(current["end"], end)
        current["text"] = _append_text(current["text"], word["text"])
        current["overlap"] = current["overlap"] or bool(word.get("overlap", False))
    return segments


def format_timestamp(seconds, srt=False):
    milliseconds = max(0, round(float(seconds) * 1000))
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    separator = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}"


def _segment_text(segment, diarize, speaker_names):
    text = segment["text"]
    if diarize:
        speaker = segment.get("speaker")
        label = "Не определён"
        if speaker is not None:
            label = f"Говорящий {speaker}"
            if 0 < speaker <= len(speaker_names) and speaker_names[speaker - 1]:
                label = speaker_names[speaker - 1]
        text = f"{label}: {text}"
    if segment.get("overlap", False):
        text += " [перекрытие голосов]"
    return text


def render_txt(segments, diarize=True, speaker_names=()):
    lines = [
        f"[{format_timestamp(segment['start'])}] {_segment_text(segment, diarize, speaker_names)}"
        for segment in segments
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def render_srt(segments, diarize=True, speaker_names=()):
    blocks = []
    for index, segment in enumerate(segments, start=1):
        timing = f"{format_timestamp(segment['start'], srt=True)} --> {format_timestamp(segment['end'], srt=True)}"
        blocks.append(f"{index}\n{timing}\n{_segment_text(segment, diarize, speaker_names)}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")
