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
import android.media.audiofx.NoiseSuppressor
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.core.app.ActivityCompat
import androidx.core.app.ComponentActivity
import java.nio.ByteOrder
import kotlin.math.sin


@SuppressLint("RestrictedApi")
class TestStreaming: ComponentActivity() {
    var isRecording = false
    var am: AudioManager? = null
    var record: AudioRecord? = null
    var track: AudioTrack? = null

    val AUDIOFORMAT = AudioFormat.ENCODING_PCM_FLOAT
    var noiserOn = false

    // C++ Stuff
    object Network {
        external fun runDenoiser(input: DoubleArray): DoubleArray
        external fun init()
        external fun freeMem()
        init {
            System.loadLibrary("network")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        volumeControlStream = AudioManager.STREAM_VOICE_CALL

        Network.init()
        /*val doub = DoubleArray(128)
        for (i in 0..127)
        {
            doub[i] = sin(i * 0.1)
            Log.d("CallingC", "Input value $i: " + doub[i])
        }
        val out = Network.runDenoiser(doub)
        for (i in 0..127)
        {
            Log.d("CallingC", "Output value $i: " + out[i])
        }*/

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

    override fun onDestroy() {
        super.onDestroy()
        Network.freeMem()
    }

    private fun initRecordAndTrack() {
        ActivityCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO
        ) != PackageManager.PERMISSION_GRANTED

        val sampleRate = 44100

        val minBufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AUDIOFORMAT
        )

        //Three sources: VOICE_RECOGNITION gives smoother speech and best low latency ((1000 * 128)/44100 ~ 2.9ms) with short array (128)
        //                               but with background noise and a little bit echo when the audio is further and a little bit smaller volume.
        //             VOICE_COMMUNICATION gives cut-off speech and bigger latency due to bigger short array (2048)
        //                                  but no background noise and no echo.
        //              MIC is a better version of VOICE_COMMUNICATION but very unreliable if the audio has a big variation in volume.


        record = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AUDIOFORMAT,
            minBufferSize
        )

        if (NoiseSuppressor.isAvailable()) {
            val suppressor = NoiseSuppressor.create(record!!.audioSessionId)
            suppressor.enabled = true
        }

        if (AcousticEchoCanceler.isAvailable()) {
            val echoCanceler = AcousticEchoCanceler.create(record!!.audioSessionId)
            echoCanceler.enabled = true
        }

        val maxJitter = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AUDIOFORMAT
        )

        track = AudioTrack(
            AudioManager.STREAM_MUSIC,
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AUDIOFORMAT,
            maxJitter,
            AudioTrack.PERFORMANCE_MODE_LOW_LATENCY,
            record!!.audioSessionId
        )
    }

    private fun recordAndPlay() {
        // The smaller the size of array, the less latency it is. This is the array for audio data in queue.
        // Access this array when manipulating data. The data is in PCM ENCODING 16 bits audio, not .wav
        val lin = FloatArray(128)
        val doub = DoubleArray(128)
        var out = DoubleArray(128)
        var num = 0
        am!!.mode = AudioManager.MODE_IN_COMMUNICATION
        while (true) {
            if (isRecording) {
                num = record!!.read(lin, 0, 128, AudioRecord.READ_NON_BLOCKING)

                if (noiserOn)
                {
                    // Denoiser run
                    //Log.d("CallingC", "Raw input: " + lin[0] + " / " + lin[0].toDouble())
                    for (i in 0..127)
                    {
                        doub[i] = (lin[i].toDouble())
                    }
                    //Log.d("CallingC", "Converted input: " + doub[0])

                    // Run through the denoiser
                    out = Network.runDenoiser(doub)

                    //Log.d("CallingC", "Raw out: " + out[0])
                    for (i in 0..127)
                    {
                        lin[i] = ((out[i]).toFloat()) * 0.1f + 0.9f * lin[i]
                    }
                    //Log.d("CallingC", "Converted out: " + lin[0])
                    //Log.d("CallingC","Endian Order: " + ByteOrder.nativeOrder())
                }



                track!!.write(lin, 0, num, AudioTrack.WRITE_NON_BLOCKING)
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