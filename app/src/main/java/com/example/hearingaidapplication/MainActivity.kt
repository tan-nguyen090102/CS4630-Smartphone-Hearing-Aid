package com.example.hearingaidapplication
import android.Manifest
import android.graphics.Color
import android.media.AudioFormat
import android.media.MediaPlayer
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.SeekBar
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.core.app.ActivityCompat
import com.github.squti.androidwaverecorder.WaveRecorder
import com.masoudss.lib.WaveformSeekBar
import com.masoudss.lib.utils.WaveGravity
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.Runnable
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    //waveform seekbar: https://github.com/massoudss/waveformSeekBar
    //seekbar: https://www.geeksforgeeks.org/seekbar-in-kotlin/#

    private lateinit var waveformSeekBar: WaveformSeekBar
    private lateinit  var filePath: String
    private var handler: Handler = Handler(Looper.myLooper()!!)
    private lateinit var runnable: Runnable

    /**Set up output wav file*/


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        /**Set Permission*/
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.RECORD_AUDIO),
            0
        )

        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE),
            0
        )

        /**Find view*/
        val seek = findViewById<SeekBar>(R.id.seekBar)
        waveformSeekBar = findViewById(R.id.waveformSeekBar)
        val startButton = findViewById<Button>(R.id.startButton)
        val stopButton = findViewById<Button>(R.id.stopButton)


        /**This path points to application cache directory*/
        //Path: Internal Storage/Android/data/com.example.hearingaidapplication/cache
        filePath = externalCacheDir?.absolutePath + "/audio.wav"
        val waveRecorder = WaveRecorder(filePath)

        /**Set up player*/
        val player = MediaPlayer()
        player.isLooping = false

        //applicationContext.cacheDir?.deleteRecursively()

        /**Config waveRecorder*/
        waveRecorder.waveConfig.sampleRate = 44100
        waveRecorder.waveConfig.channels = AudioFormat.CHANNEL_IN_STEREO
        waveRecorder.waveConfig.audioEncoding = AudioFormat.ENCODING_PCM_16BIT
        waveRecorder.noiseSuppressorActive = true

        /**Initial run*/
        waveRecorder.startRecording()
        //Threshold: 200 ms delay and 300 ms postDelay
        GlobalScope.launch {
            delay(500)
            waveRecorder.stopRecording()
        }

        /**Looping run*/
        runnable = object : Runnable {
            override fun run() {
                player.reset()
                player.setDataSource(filePath)
                player.prepare()
                waveRecorder.startRecording()
                //Threshold: 200 ms delay and 300 ms postDelay
                //If we play at minimum threshold, the audio is unrecognizable.
                //Threshold 1000 ms and 1100 ms respectively gives an okay experience though being delayed.
                GlobalScope.launch {
                    delay(1000)
                    waveRecorder.stopRecording()
                    //This function may delay the sound so commented for now.
                    //waveformSeekBar.setSampleFrom(filePath)
                }
                player.start()
                handler.postDelayed(this, 1100)
            }
        }

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
        }
    }
}