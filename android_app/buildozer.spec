[app]

title = Kerala IT Hub
package.name = keralaithub
package.domain = org.geniai

source.dir = .
source.include_exts = py,png,jpg,kv,atlas

version = 1.0.0

requirements = python3,kivy==2.3.1,requests,certifi,urllib3,idna,charset-normalizer

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

[buildozer]

log_level = 2
warn_on_root = 1
