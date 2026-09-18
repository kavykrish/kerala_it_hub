# Deploying Kerala IT Hub as an Android App

## How it fits together

The RAG pipeline (web search, page scraping, chunking, embeddings, Groq)
is too heavy to run on a phone, so the architecture is a thin Android
client talking to a backend you host separately:

```
Android app (Kivy)  --HTTPS-->  FastAPI backend  -->  MCP server  -->  Groq
     (android_app/)                (backend/)      (mcp_server/)
```

You deploy the backend once to a public HTTPS URL, then build the
Android app pointed at that URL.

---

## 1. Deploy the backend

### Option A: Render.com (recommended for a free demo)

1. Push this repo to GitHub (see step 0 below if you haven't yet).
2. On [render.com](https://render.com), click **New +** -> **Blueprint**,
   and point it at your repo. Render will detect `render.yaml`
   automatically and configure the service.
3. When prompted for environment variables, set:
   - `GROQ_API_KEY` -- your Groq key (rotate the one currently in your
     local `.env` before going live, since free-tier keys are easy to
     leak by accident -- generate a fresh one at
     [console.groq.com](https://console.groq.com/keys) and use that).
   - `API_KEY` -- any random string of your choosing. This becomes the
     shared secret the Android app must send; it stops strangers who
     get a copy of your APK from burning through your Groq quota.
4. Deploy. First build takes several minutes (installing torch +
   sentence-transformers). Render gives you a URL like
   `https://kerala-it-hub-api.onrender.com`.
5. Test it:
   ```
   curl https://kerala-it-hub-api.onrender.com/health
   ```

**Free-tier caveat:** the service spins down after 15 minutes of
inactivity, so the first request after idling can take 30-60 seconds
while it wakes up (the app will just show "Searching..." for longer --
it isn't broken).

RAM is capped at 512MB on the free tier. The backend originally used
`sentence-transformers` (PyTorch-based), which reliably went **over**
512MB on import alone and crashed every deploy with `Out of memory
(used over 512Mi)`. It now uses `fastembed` (ONNX runtime) instead --
same embedding model (`all-MiniLM-L6-v2`), same retrieval quality, but
real measured usage is ~90MB for the FastAPI process and ~280MB for
the MCP worker that holds the model, comfortably under the limit. If
you still see OOM crashes in the Logs tab (e.g. from many concurrent
requests), upgrade to the paid Starter plan ($7/mo, 2GB RAM).

### Option B: Your own server/VPS

Use the included `Dockerfile`:

```bash
docker build -t kerala-it-hub .
docker run -p 8000:8000 --env-file .env kerala-it-hub
```

Put a reverse proxy (nginx/Caddy) in front for HTTPS (e.g. Caddy with
automatic Let's Encrypt certs is the least setup).

### A note on the MCP subprocess

The backend used to start a brand-new MCP subprocess (and reload the
embedding model) on every single question -- that's now fixed:
`backend/main.py` starts the MCP session once at boot and reuses it.
One consequence worth knowing: that stdio session only handles one
request at a time, so concurrent questions are queued rather than
processed in parallel. Fine for a class project's traffic; if you ever
need real concurrency, that's the place to revisit (e.g. a small pool
of MCP sessions).

---

## 2. Point the Android app at your backend

Open [android_app/main.py](android_app/main.py) and change:

```python
DEFAULT_API_BASE_URL = "https://YOUR-BACKEND-URL.onrender.com"
DEFAULT_API_KEY = ""  # the same value you set as API_KEY on Render
```

You don't have to rebuild the APK every time you change hosts, though
-- there's a **Settings** button in the app's top bar where the backend
URL and API key can be edited and saved on-device at runtime.

To test against your backend from a desktop first (no build needed):

```bash
python -m pip install kivy requests certifi
python android_app/main.py
```

---

## 3. Build the APK (GitHub Actions -- no WSL needed)

`.github/workflows/build-apk.yml` is already set up. Once this repo is
on GitHub:

1. Go to the **Actions** tab -> **Build Android APK** -> **Run workflow**.
2. Wait for it to finish (first run is slow -- it downloads the Android
   SDK/NDK; ~15-20 minutes).
3. Open the finished run and download the `kerala-it-hub-debug-apk`
   artifact -- that's your installable APK.
4. Copy it to your phone and install it (you'll need to allow "install
   from unknown sources" for a debug build that isn't from the Play
   Store).

The workflow also re-runs automatically whenever you push changes
under `android_app/`.

If you'd rather build locally on this Windows machine, you need WSL2
(Buildozer only runs on Linux):

```bash
wsl --install
# inside the WSL Ubuntu shell:
sudo apt update && sudo apt install -y python3-pip build-essential git \
    openjdk-17-jdk unzip zlib1g-dev libncurses5
pip install buildozer cython
cd android_app
buildozer android debug
```

The APK lands in `android_app/bin/`.

---

## 4. Step 0: if this isn't a git repo yet

```bash
git init
git add .
git commit -m "Initial commit"
```

Then create a GitHub repo and push. **Before your first push**, double
check `.env` is not staged (`git status` -- it should be ignored
already per `.gitignore`) since it holds your Groq key.

---

## 5. Recommended test order

1. Run the backend locally (`python -m backend.main`) and confirm
   `curl http://127.0.0.1:8000/health` works.
2. Point the desktop Kivy app at `http://127.0.0.1:8000` via Settings
   and ask a real question end-to-end.
3. Deploy the backend (step 1) and confirm `/health` on the public URL.
4. Point the app at the public URL and re-test the same question.
5. Build and install the APK (step 3) and test on an actual phone over
   mobile data (not just WiFi) to make sure nothing was relying on
   being on the same network as your PC.
