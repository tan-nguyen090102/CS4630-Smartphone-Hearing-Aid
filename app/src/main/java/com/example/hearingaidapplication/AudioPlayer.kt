package com.example.hearingaidapplication
import java.io.File


interface AudioPlayer {
    fun playFile(filePath: String)
    fun stop()

}