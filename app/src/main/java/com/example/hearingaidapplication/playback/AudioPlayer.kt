package com.example.hearingaidapplication.playback

import java.io.File
interface AudioPlayer {
    fun playFile(file: File)
    fun stop()
}