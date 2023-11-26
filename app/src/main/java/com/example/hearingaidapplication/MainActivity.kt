package com.example.hearingaidapplication
import android.Manifest
import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Menu
import android.view.View
import android.widget.Button
import androidx.activity.ComponentActivity
import androidx.core.app.ActivityCompat
import com.masoudss.lib.WaveformSeekBar
import kotlinx.coroutines.Runnable
import kotlin.math.log


class MainActivity : ComponentActivity() {

    //waveform seekbar: https://github.com/massoudss/waveformSeekBar
    //seekbar: https://www.geeksforgeeks.org/seekbar-in-kotlin/#

    private lateinit var waveformSeekBar: WaveformSeekBar
    private lateinit  var filePath: String
    private var handler: Handler = Handler(Looper.myLooper()!!)
    private lateinit var runnable: Runnable

    /**Set up output wav file*/
    private lateinit var audioManager: AudioManager
    private lateinit var audioRecord: AudioRecord
    private lateinit var audioTrack: AudioTrack

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //setContentView(R.layout.activity_main)
        setContentView(R.layout.activity_record)

        /**Setup audio*/
        volumeControlStream = AudioManager.MODE_IN_COMMUNICATION
        init()
        object : Thread() {
            override fun run() {
                recordAndPlay()
                Log.d("Dualsession","insession")
            }
        }.start()

        /**Set Permission*/
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

        /*
        /**Find view*/
        val seek = findViewById<SeekBar>(R.id.seekBar)
        waveformSeekBar = findViewById(R.id.waveformSeekBar)
        val startButton = findViewById<Button>(R.id.startButton)
        val stopButton = findViewById<Button>(R.id.stopButton)




        /**Button Listeners*/
        startButton.setOnClickListener {
            Toast.makeText(applicationContext, "Start recording", Toast.LENGTH_SHORT).show()
            handler.post(runnable)
        }

        stopButton.setOnClickListener {
            Toast.makeText(applicationContext, "Stop recording", Toast.LENGTH_SHORT).show()
            handler.removeCallbacks(runnable)
        }

        /**Set up seek bar*/
        seek?.setOnSeekBarChangeListener(object :
            SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(
                seek: SeekBar,
                progress: Int, fromUser: Boolean
            ) {
            }

            override fun onStartTrackingTouch(p0: SeekBar?) {
                TODO("Not yet implemented")
            }

            override fun onStopTrackingTouch(p0: SeekBar?) {
                TODO("Not yet implemented")
            }
        })

        /**Set up waveform display*/
        waveformSeekBar.apply {
            maxProgress = 30F
            progress = 33.2F
            waveWidth = 5F
            waveGap = 5F
            waveMinHeight = 5F
            waveCornerRadius = 10F
            waveGravity = WaveGravity.CENTER
            wavePaddingTop = 2
            wavePaddingBottom = 2
            wavePaddingRight = 1
            wavePaddingLeft = 1
            waveBackgroundColor = Color.GRAY
            waveProgressColor = Color.BLUE
            markerTextColor = Color.MAGENTA
            markerTextPadding = 1F
            markerTextSize = 12F
            markerWidth = 1F
            markerColor = Color.RED
        }*/
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.main, menu)
        return true
    }

    private fun init() {
        ActivityCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO
        )
        audioManager = this.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        var min = AudioRecord.getMinBufferSize(44100, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        audioRecord = AudioRecord(MediaRecorder.AudioSource.VOICE_COMMUNICATION, 44100, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, min)
        var maxJitter = AudioTrack.getMinBufferSize(44100, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val audioAttributes = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
            .build()
        val audioFormat = AudioFormat.Builder()
            .setSampleRate(44100)
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
            .build()

        audioTrack = AudioTrack(audioAttributes,
            audioFormat, maxJitter, AudioTrack.MODE_STREAM, 0)
    }

    private fun recordAndPlay() {
        var lin = shortArrayOf(1024)
        var num = 0
        audioManager = this.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
        audioRecord.startRecording()
        audioTrack.play()
        while (true) {
            num = audioRecord.read(lin, 0, 1024)
            audioTrack.write(lin, 0, num)
        }
    }

    var isSpeaker = false

    fun modeChange(view: View) {
        var modeButton = findViewById<Button>(R.id.modeBtn)
        if (isSpeaker == true) {
            audioManager.isSpeakerphoneOn = false
            isSpeaker = false
            modeButton.text = "Call mode"
        } else {
            audioManager.isSpeakerphoneOn = true
            isSpeaker = true
            modeButton.text = "Speaker mode"
        }
    }

    var isPlaying = true

    fun play(view: View) {
        var playButton = findViewById<Button>(R.id.playBtn)
        if (isPlaying) {
            audioRecord.stop()
            audioTrack.pause()
            isPlaying = false
            playButton.text = "Play"
        } else {
            audioRecord.startRecording()
            audioTrack.play()
            isPlaying = true
            playButton.text = "Pause"
        }
    }
}


