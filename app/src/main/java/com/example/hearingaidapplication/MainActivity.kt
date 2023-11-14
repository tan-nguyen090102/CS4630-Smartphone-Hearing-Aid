package com.example.hearingaidapplication
import android.graphics.Color
import android.os.Bundle
import android.widget.SeekBar
import android.widget.Toast
import androidx.activity.ComponentActivity
import com.masoudss.lib.SeekBarOnProgressChanged
import java.io.File
import com.masoudss.lib.WaveformSeekBar
import com.masoudss.lib.utils.WaveGravity
import android.Manifest
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.Button
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.core.app.ActivityCompat
import com.example.hearingaidapplication.ui.theme.HearingAidApplicationTheme
import com.github.squti.androidwaverecorder.WaveRecorder
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.Runnable
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.Timer

class MainActivity : ComponentActivity() {

    //waveform seekbar: https://github.com/massoudss/waveformSeekBar
    //seekbar: https://www.geeksforgeeks.org/seekbar-in-kotlin/#

    private lateinit var waveformSeekBar: WaveformSeekBar
    private lateinit  var filePath: String
    private var handler: Handler = Handler(Looper.myLooper()!!)
    private lateinit var runnable: Runnable

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
        runnable = object : Runnable {
            override fun run() {
                waveRecorder.startRecording()
                waveRecorder.stopRecording()
                handler.postDelayed(this, 500)
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
            override fun onProgressChanged(seek: SeekBar,
                                           progress: Int, fromUser: Boolean) {
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
            progress = 33.2F
            waveWidth = 10F
            waveGap = 20F
            waveMinHeight = 5F
            waveCornerRadius = 10F
            waveGravity = WaveGravity.CENTER
            wavePaddingTop = 2
            wavePaddingBottom = 2
            wavePaddingRight = 2
            wavePaddingLeft = 2
            waveBackgroundColor = Color.GRAY
            waveProgressColor = Color.BLUE
            markerTextColor = Color.MAGENTA
            markerTextPadding = 1F
            markerTextSize = 12F
            markerWidth = 1F
            markerColor = Color.RED
        }
        waveformSeekBar.setSampleFrom(filePath)


    }

    /**Button listeners*/


    //The function that record the audio and turn to wav file in a specific time interval. Return a filepath containing that audio file.
    private fun intervalRecording(time: Long): String {
        return filePath;
    }

}


