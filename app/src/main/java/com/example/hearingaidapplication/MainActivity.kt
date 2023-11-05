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

class MainActivity : ComponentActivity() {

    //waveform seekbar: https://github.com/massoudss/waveformSeekBar
    //seekbar: https://www.geeksforgeeks.org/seekbar-in-kotlin/#

    private lateinit var waveformSeekBar: WaveformSeekBar
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val seek = findViewById<SeekBar>(R.id.seekBar)
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
        waveformSeekBar = findViewById(R.id.waveformSeekBar)

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

        waveformSeekBar.setSampleFrom(R.raw.audio)

        /**
         * This path points to application cache directory.
         * you could change it based on your usage
         */
        //Path: Internal Storage/Android/data/com.example.hearingaidapplication/cache
        val filePath:String = externalCacheDir?.absolutePath + "/audio.wav"
        Toast.makeText(this, filePath, Toast.LENGTH_LONG).show()
        val waveRecorder = WaveRecorder(filePath)

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

        setContent {
            HearingAidApplicationTheme {
                Column(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Button(onClick = {
                        waveRecorder.startRecording()
                    }) {
                        Text(text = "Start recording")
                    }
                    Button(onClick = {
                        waveRecorder.stopRecording()
                    }) {
                        Text(text = "Stop recording")
                    }
                }
            }
        }
    }
}


