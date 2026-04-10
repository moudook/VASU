#!/usr/bin/env python3
"""
VASU - Tool Handler
Executes tool calls from LLM output: camera, alarm, notes, web search, devices, time, weather.
"""

import datetime
import json
import logging
import os
import re
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

NOTES_DIR = "/var/vasu/notes"
ALARMS_FILE = "/etc/vasu/alarms.json"


class ToolHandler:
    """Parses and executes tool calls from LLM responses."""

    def try_execute(self, response):
        """Check if response contains a tool call and execute it."""
        tool_call = self._parse_tool_call(response)
        if tool_call is None:
            return None

        tool_name = tool_call.get("tool", "")
        params = tool_call.get("params", {})

        log.info("Tool call: %s(%s)", tool_name, params)

        handlers = {
            "invoke_camera": self._invoke_camera,
            "set_alarm": self._set_alarm,
            "add_note": self._add_note,
            "web_search": self._web_search,
            "toggle_device": self._toggle_device,
            "get_time": self._get_time,
            "get_weather": self._get_weather,
        }

        handler = handlers.get(tool_name)
        if handler is None:
            log.warning("Unknown tool: %s", tool_name)
            return {"error": "Unknown tool: " + tool_name}

        try:
            result = handler(params)
            result["needs_followup"] = tool_name in ("web_search", "invoke_camera")
            return result
        except Exception as e:
            log.error("Tool execution failed: %s", e)
            return {"error": str(e)}

    def _parse_tool_call(self, response):
        """Extract tool call JSON from LLM response."""
        # Try to find {"tool": "...", "params": {...}} pattern
        pattern = r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"params"\s*:\s*(\{[^}]*\})\s*\}'
        match = re.search(pattern, response)
        if match:
            try:
                tool_name = match.group(1)
                params = json.loads(match.group(2))
                return {"tool": tool_name, "params": params}
            except json.JSONDecodeError:
                pass

        # Try full JSON parse
        json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def _invoke_camera(self, params):
        """Capture image with libcamera."""
        query = params.get("query", "describe what you see")
        camera = params.get("camera", "rear")
        cam_id = "0" if camera == "rear" else "1"
        image_path = "/tmp/capture.jpg"

        try:
            subprocess.run(
                ["libcamera-still", "-o", image_path, "--immediate",
                 "--timeout", "1000", "--camera", cam_id],
                timeout=5, capture_output=True, check=True,
            )
            return {
                "status": "captured",
                "image_path": image_path,
                "query": query,
                "needs_vlm": True,
            }
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            return {"status": "error", "error": "Camera failed: " + str(e)}

    def _set_alarm(self, params):
        """Set an alarm using systemd timer."""
        alarm_time = params.get("time", "07:00")
        label = params.get("label", "Alarm")
        days = params.get("days", [])

        # Load existing alarms
        alarms = []
        if os.path.exists(ALARMS_FILE):
            try:
                with open(ALARMS_FILE) as f:
                    alarms = json.load(f)
            except Exception:
                alarms = []

        alarm_entry = {
            "time": alarm_time,
            "label": label,
            "days": days,
            "created": datetime.datetime.now().isoformat(),
            "active": True,
        }
        alarms.append(alarm_entry)

        os.makedirs(os.path.dirname(ALARMS_FILE), exist_ok=True)
        with open(ALARMS_FILE, "w") as f:
            json.dump(alarms, f, indent=2)

        # Create systemd timer for the alarm
        try:
            timer_name = "vasu-alarm-%s" % alarm_time.replace(":", "")
            service_content = (
                "[Unit]\nDescription=Vasu Alarm: %s\n\n"
                "[Service]\nType=oneshot\n"
                "ExecStart=/usr/bin/aplay /opt/vasu/sounds/alarm.wav\n"
            ) % label

            timer_content = (
                "[Unit]\nDescription=Vasu Alarm Timer: %s\n\n"
                "[Timer]\nOnCalendar=*-*-* %s:00\nPersistent=true\n\n"
                "[Install]\nWantedBy=timers.target\n"
            ) % (label, alarm_time)

            service_path = "/etc/systemd/system/%s.service" % timer_name
            timer_path = "/etc/systemd/system/%s.timer" % timer_name

            Path(service_path).write_text(service_content)
            Path(timer_path).write_text(timer_content)

            subprocess.run(["systemctl", "daemon-reload"], capture_output=True)
            subprocess.run(["systemctl", "enable", "--now", timer_name + ".timer"],
                           capture_output=True)
        except Exception as e:
            log.warning("Systemd timer setup failed: %s", e)

        return {"status": "alarm_set", "time": alarm_time, "label": label}

    def _add_note(self, params):
        """Save a note to the daily notes file."""
        content = params.get("content", "")
        title = params.get("title", "Note")

        if not content:
            return {"status": "error", "error": "Empty note"}

        os.makedirs(NOTES_DIR, exist_ok=True)
        today = datetime.date.today().isoformat()
        note_file = os.path.join(NOTES_DIR, today + ".md")

        timestamp = datetime.datetime.now().strftime("%H:%M")
        entry = "\n## %s (%s)\n%s\n" % (title, timestamp, content)

        with open(note_file, "a", encoding="utf-8") as f:
            f.write(entry)

        return {"status": "note_saved", "file": note_file}

    def _web_search(self, params):
        """Search the web using DuckDuckGo HTML lite."""
        query = params.get("query", "")
        if not query:
            return {"status": "error", "error": "Empty query"}

        try:
            import urllib.request
            import urllib.parse

            url = "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(query)
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Linux; Android) Vasu/1.0"
            })
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode("utf-8", errors="ignore")

            # Simple HTML text extraction
            import html as html_module
            text = re.sub(r'<[^>]+>', ' ', html)
            text = html_module.unescape(text)
            text = re.sub(r'\s+', ' ', text).strip()

            # Take first 500 chars of useful content
            summary = text[:500]
            return {"status": "search_result", "query": query, "result": summary}

        except Exception as e:
            return {"status": "error", "error": "Search failed: " + str(e)}

    def _toggle_device(self, params):
        """Toggle a smart home device via MQTT or HTTP."""
        device = params.get("device", "")
        state = params.get("state", "on")

        if not device:
            return {"status": "error", "error": "No device specified"}

        # Try MQTT first
        try:
            topic = "home/devices/%s/set" % device.lower().replace(" ", "_")
            payload = json.dumps({"state": state.upper()})
            subprocess.run(
                ["mosquitto_pub", "-t", topic, "-m", payload, "-h", "localhost"],
                timeout=3, capture_output=True,
            )
            return {"status": "toggled", "device": device, "state": state}
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback: log the action
        log.info("Device toggle (no MQTT): %s -> %s", device, state)
        return {"status": "toggled", "device": device, "state": state,
                "note": "MQTT not available, logged only"}

    def _get_time(self, params):
        """Get current time."""
        now = datetime.datetime.now()
        return {
            "status": "time",
            "time": now.strftime("%H:%M"),
            "date": now.strftime("%Y-%m-%d"),
            "day": now.strftime("%A"),
            "formatted": now.strftime("%A, %d %B %Y, %I:%M %p"),
        }

    def _get_weather(self, params):
        """Get weather from wttr.in (no API key needed)."""
        location = params.get("location", "Delhi")

        try:
            import urllib.request
            url = "https://wttr.in/%s?format=j1" % location.replace(" ", "+")
            req = urllib.request.Request(url, headers={
                "User-Agent": "Vasu/1.0"
            })
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))

            current = data.get("current_condition", [{}])[0]
            return {
                "status": "weather",
                "location": location,
                "temp_c": current.get("temp_C", "?"),
                "feels_like_c": current.get("FeelsLikeC", "?"),
                "humidity": current.get("humidity", "?"),
                "description": current.get("weatherDesc", [{}])[0].get("value", "?"),
                "wind_kmph": current.get("windspeedKmph", "?"),
            }
        except Exception as e:
            return {"status": "error", "error": "Weather failed: " + str(e)}
