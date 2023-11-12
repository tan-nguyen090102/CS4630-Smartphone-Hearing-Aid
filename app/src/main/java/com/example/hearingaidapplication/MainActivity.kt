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
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

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
                        Toast.makeText(applicationContext, "Start recording", Toast.LENGTH_SHORT).show()
                        var filePath = intervalRecording(500)
                    }) {
                        Text(text = "Start recording")
                    }
                }
            }
        }
    }

    //The function that record the audio and turn to wav file in a specific time interval. Return a filepath containing that audio file.
    private fun intervalRecording(time: Long): String {
        /**
         * This path points to application cache directory.
         * you could change it based on your usage
         */
        //Path: Internal Storage/Android/data/com.example.hearingaidapplication/cache
        val filePath:String = externalCacheDir?.absolutePath + "/audio.wav"
        val waveRecorder = WaveRecorder(filePath)

        waveRecorder.startRecording()
        GlobalScope.launch {
            delay(time)
            waveRecorder.stopRecording()
        }
        return filePath;
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