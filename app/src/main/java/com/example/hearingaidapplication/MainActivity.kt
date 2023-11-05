package com.example.hearingaidapplication

import android.Manifest
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
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

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    HearingAidApplicationTheme {
        Greeting("Android")
    }
}