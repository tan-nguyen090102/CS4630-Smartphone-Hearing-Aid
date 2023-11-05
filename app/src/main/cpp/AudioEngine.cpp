#include <android/log.h>
#include "AudioEngine.h"
#include <thread>
#include <mutex>

//Double-buffering offers a good tradeoff between latency and protection against glitches
constexpr int32_t kBufferSizeInBursts = 2;

aaudio_data_callback_result_t dataCallback(
        AAudioStream *stream,
        void *userData,
        void *audioData,
        int32_t numFrames) {

    ((Oscillator *) (userData)) ->render(static_cast<float *>(audioData), numFrames);

    return AAUDIO_CALLBACK_RESULT_CONTINUE;
}

void errorCallback(
        AAudioStream *stream,
        void *userData,
        aaudio_result_t error) {

    if (error == AAUDIO_ERROR_DISCONNECTED) {
        std::function<void(void)> restartFunction = std::bind(&AudioEngine::restart, static_cast<AudioEngine *>(userData));

        new std::thread(restartFunction);
    }
}

bool AudioEngine::start() {
    //Create stream builder object
    AAudioStreamBuilder *streamBuilder;
    AAudio_createStreamBuilder(&streamBuilder);

    //Add parameters to stream builder
    AAudioStreamBuilder_setFormat(streamBuilder, AAUDIO_FORMAT_PCM_FLOAT);
    AAudioStreamBuilder_setChannelCount(streamBuilder, 1);
    AAudioStreamBuilder_setPerformanceMode(streamBuilder, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
    AAudioStreamBuilder_setDataCallback(streamBuilder, ::dataCallback, &oscillator_);
    AAudioStreamBuilder_setErrorCallback(streamBuilder, ::errorCallback, this);

    //Open the stream
    aaudio_result_t result = AAudioStreamBuilder_openStream(streamBuilder, &stream_);
    if (result != AAUDIO_OK) {
        __android_log_print(ANDROID_LOG_ERROR, "AudioEngine", "Error opening stream %s",
                            AAudio_convertResultToText((result)));
        return false;
    }

    //Retrieves the sample rate of the stream for the oscillator.
    int32_t sampleRate = AAudioStream_getSampleRate(stream_);
    oscillator_.setSampleRate(sampleRate);

    //Set the buffer size
    AAudioStream_setBufferSizeInFrames(stream_, AAudioStream_getFramesPerBurst(stream_) * kBufferSizeInBursts);

    //Start the stream
    result = AAudioStream_requestStart(stream_);
    if (result != AAUDIO_OK) {
        __android_log_print(ANDROID_LOG_ERROR, "AudioEngine", "Error starting stream %s",
                            AAudio_convertResultToText((result)));
        return false;
    }

    AAudioStreamBuilder_delete(streamBuilder);
    return true;
}

void AudioEngine::restart() {
    static std::mutex restartingLock;
    if (restartingLock.try_lock()) {
        stop();
        start();
        restartingLock.unlock();
    }
}

void AudioEngine::stop() {
    if (stream_ != nullptr) {
        AAudioStream_requestStop(stream_);
        AAudioStream_close(stream_);
    }
}

void AudioEngine::setToneOn(bool isToneOn) {
    oscillator_.setWaveOn(isToneOn);
}