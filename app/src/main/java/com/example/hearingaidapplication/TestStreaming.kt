package com.example.hearingaidapplication

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.os.Bundle
import android.widget.Button
import androidx.core.app.ActivityCompat
import androidx.core.app.ComponentActivity


@SuppressLint("RestrictedApi")
class TestStreaming: ComponentActivity() {
    var isRecording = false
    var am: AudioManager? = null
    var record: AudioRecord? = null
    var track: AudioTrack? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        volumeControlStream = AudioManager.MODE_IN_COMMUNICATION

        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.RECORD_AUDIO),
            0
        )

        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.MODIFY_AUDIO_SETTINGS),
            0
        )

        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE),
            0
        )

        initRecordAndTrack()
        am = this.getSystemService(Context.AUDIO_SERVICE) as AudioManager?
        am!!.isSpeakerphoneOn = false
        object : Thread() {
            override fun run() {
                recordAndPlay()
            }
        }.start()

        val startButton = findViewById<Button>(R.id.startButton)
        startButton.setOnClickListener {
            if (!isRecording) {
                startRecordAndPlay()
            }
        }
        val stopButton = findViewById<Button>(R.id.stopButton)
        stopButton.setOnClickListener {
            if (isRecording) {
                stopRecordAndPlay()
            }
        }
    }

    private fun initRecordAndTrack() {
        ActivityCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO
        ) != PackageManager.PERMISSION_GRANTED

        val rate = 16000

        val min = AudioRecord.getMinBufferSize(
            rate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        record = AudioRecord(
            MediaRecorder.AudioSource.VOICE_COMMUNICATION,
            rate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            min
        )

        if (AcousticEchoCanceler.isAvailable()) {
            val echoCanceler = AcousticEchoCanceler.create(record!!.audioSessionId)
            echoCanceler.enabled = true
        }
        val maxJitter = AudioTrack.getMinBufferSize(
            rate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        track = AudioTrack(
            AudioManager.MODE_IN_COMMUNICATION,
            rate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxJitter,
            AudioTrack.PERFORMANCE_MODE_LOW_LATENCY
        )
    }

    private fun recordAndPlay() {
        val lin = ShortArray(2048)
        var num = 0
        am!!.mode = AudioManager.MODE_IN_COMMUNICATION
        while (true) {
            if (isRecording) {
                num = record!!.read(lin, 0, 2048)
                track!!.write(lin, 0, num)
            }
        }
    }

    private fun startRecordAndPlay() {
        record!!.startRecording()
        track!!.play()
        isRecording = true
    }

    private fun stopRecordAndPlay() {
        record!!.stop()
        track!!.pause()
        isRecording = false
    }

}