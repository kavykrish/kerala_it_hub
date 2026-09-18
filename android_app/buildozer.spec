[app]

title = Kerala IT Hub
package.name = keralaithub
package.domain = org.geniai

source.dir = .
source.include_exts = py,png,jpg,kv,atlas

version = 1.0.0

# charset-normalizer>=3.5.0 publishes Android-tagged wheels that
# python-for-android's pip invocation can't actually use (it doesn't
# pass the cross-compile flags needed to accept them), so pip rejects
# them as "not a supported wheel on this platform" and the build dies.
# Pinning below 3.5.0 keeps it on the sdist path, which p4a handles fine.
requirements = python3,kivy==2.3.1,requests,certifi,urllib3,idna,charset-normalizer<3.5.0

icon.filename = %(source.dir)s/assets/icon.png
presplash.filename = %(source.dir)s/assets/presplash.png

orientation = portrait
fullscreen = 0

# INTERNET is required to reach the backend API.
android.permissions = INTERNET

android.api = 34
android.minapi = 24
android.ndk = 25b
android.archs = arm64-v8a, armeabi-v7a
android.allow_backup = True

# Required on CI: there's no interactive terminal to accept the
# Android SDK license prompts, so accept them automatically.
android.accept_sdk_license = True

# The backend is served over HTTPS, so no extra cleartext-traffic
# config is required.

# The python-for-android release bundled with buildozer==1.5.0 has a
# bug where it self-upgrades pip inside a reused build venv; if that
# upgrade is ever interrupted (e.g. a previous CI run failing mid-way),
# the venv is left with a mixed-version pip install and every later
# build fails with "ImportError: cannot import name
# 'BuildDependencyInstallError' from 'pip._internal.exceptions'"
# (https://github.com/kivy/python-for-android/issues/3364). The fix
# (--clear the venv, stop self-upgrading pip) landed in p4a's develop
# branch (PR #3360) but hasn't made it into a tagged release yet, so
# pin to a known-good develop commit until it does.
p4a.branch = develop
p4a.commit = e772ad93f20a61c0bbe1cf8955e073cfb41062e1

[buildozer]

log_level = 2
warn_on_root = 1
