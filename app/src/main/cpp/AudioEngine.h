#ifndef HEARING_AID_APPLICATION_AUDIOENGINE_H
#define HEARING_AID_APPLICATION_AUDIOENGINE_H
#include <aaudio/AAudio.h>
#include "Oscillator.h"

class AudioEngine {
public:
    bool start();
    void stop();
    void restart();
    void setToneOn(bool isToneOn);

private:
    Oscillator oscillator_;
    AAudioStream *stream_;
};

#endif //HEARING_AID_APPLICATION_AUDIOENGINE_H
