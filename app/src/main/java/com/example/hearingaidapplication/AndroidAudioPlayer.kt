package com.example.hearingaidapplication

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.media.MediaPlayer

class AndroidAudioPlayer(
    private val context: Context
) {

    val mediaPlayer: MediaPlayer? = null;
    fun playSound(fileName: String) {
        val mediaPlayer = MediaPlayer();
        try {
            mediaPlayer.setDataSource(fileName);
            mediaPlayer.prepare();
        } catch (e: Exception) {
            //e.printStackTrace();
        }
        mediaPlayer.start();
    }
}