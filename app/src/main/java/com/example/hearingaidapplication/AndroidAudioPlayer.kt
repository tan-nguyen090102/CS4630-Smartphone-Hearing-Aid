package com.example.hearingaidapplication

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.media.MediaPlayer

class AndroidAudioPlayer(
    private val context: Context
) {

    val mediaPlayer: MediaPlayer? = null;
    fun playSound(context: Context, fileName: String) {
        val mediaPlayer = MediaPlayer();
        try {
            val afd: AssetFileDescriptor = context.assets.openFd(fileName);
            mediaPlayer.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length);
            afd.close();
            mediaPlayer.prepare();
        } catch (e: Exception) {
            //e.printStackTrace();
        }
        mediaPlayer.start();
    }
}