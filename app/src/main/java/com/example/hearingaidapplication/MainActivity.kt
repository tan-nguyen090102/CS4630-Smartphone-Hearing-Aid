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
                // write custom code for progress is changed
            }

            override fun onStartTrackingTouch(seek: SeekBar) {
                // write custom code for progress is started
            }

            override fun onStopTrackingTouch(seek: SeekBar) {
                // write custom code for progress is stopped
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

    }
}


