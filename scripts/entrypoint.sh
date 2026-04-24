#!/bin/bash
# Suppress harmless GStreamer-CRITICAL from GLib/GStreamer version mismatch in DeepStream 6.4.
# Real errors (Fatal, Traceback) still print because they come from Python (fd 1).
exec python3 -c "
import subprocess
import sys
import os

proc = subprocess.Popen(
    ['python3', '/workspace/apps/vision_service/src/main.py'],
    stdout=sys.stdout,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
)

import threading
def pump(out, fd):
    try:
        for line in fd:
            if 'GStreamer-CRITICAL' in line or 'gst_meta_api_type_has_tag' in line:
                continue
            out.write(line)
            out.flush()
    except ValueError:
        pass

import sys
t = threading.Thread(target=pump, args=(sys.stderr, proc.stderr))
t.start()

try:
    proc.wait()
finally:
    proc.stderr.close()
    t.join()
"
