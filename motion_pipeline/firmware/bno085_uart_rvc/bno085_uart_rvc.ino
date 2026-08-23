// Combined timestamped four-sensor MLX90393 and BNO085 motion readout.
//
// MLX90393 wiring:
//   ESP32 GPIO21 -> SDA, GPIO22 -> SCL, both to PCA9548 (STEMMA QT).
//   MLX90393 sensors are on mux channels 0, 2, 5, and 7 at address 0x18.
//
// BNO085 wiring (UART-RVC mode):
//   BNO085 VIN -> ESP32 3V3
//   BNO085 GND -> ESP32 GND
//   BNO085 SDA (UART data out) -> ESP32 GPIO4
//   BNO085 P0 -> ESP32 3V3
//   Leave P1 low/default. The BNO085 is not connected to the I2C mux.
//
// The USB serial stream contains two independent, timestamped protocols:
//   FRAME,sequence,device_us,s0x,s0y,s0z,...,s3x,s3y,s3z
//   MOTION,sequence,device_us,yaw,pitch,roll,ax,ay,az
//
// The host parsers select their own packet type and ignore the other type.

#include <Wire.h>
#include <Adafruit_MLX90393.h>
#include <Adafruit_BNO08x_RVC.h>
#include <math.h>

constexpr uint8_t MUX_ADDR = 0x70;
constexpr uint8_t SDA_PIN = 21;
constexpr uint8_t SCL_PIN = 22;
constexpr uint8_t MLX90393_ADDR = 0x18;
constexpr uint8_t SENSOR_COUNT = 4;
constexpr uint8_t MUX_CHANNELS[SENSOR_COUNT] = {0, 2, 5, 7};
constexpr int8_t BNO_RX_PIN = 4;
constexpr int8_t BNO_TX_PIN = -1;  // UART-RVC is output-only for this pipeline.
constexpr uint32_t SERIAL_BAUD = 115200;
constexpr uint32_t SAMPLE_RATE_HZ = 50;
constexpr uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;
constexpr uint32_t SENSOR_RETRY_PERIOD_US = 1000000UL;

Adafruit_MLX90393 sensors[SENSOR_COUNT];
Adafruit_BNO08x_RVC rvc;
HardwareSerial bnoSerial(1);

bool sensorReady[SENSOR_COUNT] = {false, false, false, false};
uint32_t nextSensorRetryUs[SENSOR_COUNT] = {0, 0, 0, 0};
uint32_t fingerSequenceNumber = 0;
uint32_t motionSequenceNumber = 0;
uint32_t nextFingerSampleUs = 0;
bool motionReady = false;

void muxSelectChannel(uint8_t channel) {
  Wire.beginTransmission(MUX_ADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

bool devicePresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

bool initializeSensor(uint8_t sensorIndex) {
  const uint8_t channel = MUX_CHANNELS[sensorIndex];
  muxSelectChannel(channel);

  if (!devicePresent(MLX90393_ADDR) ||
      !sensors[sensorIndex].begin_I2C(MLX90393_ADDR, &Wire)) {
    Serial.print(F("ERROR,sensor="));
    Serial.print(sensorIndex);
    Serial.print(F(",mux_channel="));
    Serial.print(channel);
    Serial.println(F(",not_found"));
    return false;
  }

  const bool configured =
      sensors[sensorIndex].setResolution(MLX90393_X, MLX90393_RES_16) &&
      sensors[sensorIndex].setResolution(MLX90393_Y, MLX90393_RES_16) &&
      sensors[sensorIndex].setResolution(MLX90393_Z, MLX90393_RES_16) &&
      sensors[sensorIndex].setOversampling(MLX90393_OSR_1) &&
      sensors[sensorIndex].setFilter(MLX90393_FILTER_3);
  if (!configured) {
    Serial.print(F("ERROR,sensor="));
    Serial.print(sensorIndex);
    Serial.println(F(",configuration_failed"));
    return false;
  }

  Serial.print(F("SENSOR_READY,sensor="));
  Serial.print(sensorIndex);
  Serial.print(F(",mux_channel="));
  Serial.println(channel);
  return true;
}

bool readSensor(uint8_t sensorIndex, float *x, float *y, float *z) {
  if (!sensorReady[sensorIndex]) {
    const uint32_t nowUs = micros();
    if (static_cast<int32_t>(nowUs - nextSensorRetryUs[sensorIndex]) >= 0) {
      sensorReady[sensorIndex] = initializeSensor(sensorIndex);
      nextSensorRetryUs[sensorIndex] = nowUs + SENSOR_RETRY_PERIOD_US;
    }
  }

  if (!sensorReady[sensorIndex]) {
    return false;
  }

  muxSelectChannel(MUX_CHANNELS[sensorIndex]);
  if (sensors[sensorIndex].readData(x, y, z)) {
    return true;
  }

  sensorReady[sensorIndex] = false;
  nextSensorRetryUs[sensorIndex] = micros() + SENSOR_RETRY_PERIOD_US;
  Serial.print(F("ERROR,sensor="));
  Serial.print(sensorIndex);
  Serial.println(F(",measurement_failed"));
  return false;
}

void emitFingerFrame(uint32_t sampleStartedUs) {
  float readings[SENSOR_COUNT][3];
  for (uint8_t sensorIndex = 0; sensorIndex < SENSOR_COUNT; ++sensorIndex) {
    float x = NAN;
    float y = NAN;
    float z = NAN;
    readSensor(sensorIndex, &x, &y, &z);
    readings[sensorIndex][0] = x;
    readings[sensorIndex][1] = y;
    readings[sensorIndex][2] = z;
  }

  Serial.print(F("FRAME,"));
  Serial.print(fingerSequenceNumber++);
  Serial.print(',');
  Serial.print(sampleStartedUs);
  for (uint8_t sensorIndex = 0; sensorIndex < SENSOR_COUNT; ++sensorIndex) {
    for (uint8_t axis = 0; axis < 3; ++axis) {
      Serial.print(',');
      Serial.print(readings[sensorIndex][axis], 3);
    }
  }
  Serial.println();
}

void emitMotionFrame(const BNO08x_RVC_Data &reading) {
  Serial.print(F("MOTION,"));
  Serial.print(motionSequenceNumber++);
  Serial.print(',');
  Serial.print(micros());
  Serial.print(',');
  Serial.print(reading.yaw, 3);
  Serial.print(',');
  Serial.print(reading.pitch, 3);
  Serial.print(',');
  Serial.print(reading.roll, 3);
  Serial.print(',');
  Serial.print(reading.x_accel, 3);
  Serial.print(',');
  Serial.print(reading.y_accel, 3);
  Serial.print(',');
  Serial.println(reading.z_accel, 3);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial) {
    delay(10);
  }
  delay(250);

  Wire.begin(SDA_PIN, SCL_PIN);

  bnoSerial.begin(SERIAL_BAUD, SERIAL_8N1, BNO_RX_PIN, BNO_TX_PIN);
  motionReady = rvc.begin(&bnoSerial);
  if (!motionReady) {
    // Keep the MLX finger stream alive if the BNO085 is absent.
    Serial.println(F("ERROR,bno085=not_found"));
  }

  Serial.println();
  Serial.println(F("Combined MLX90393 + BNO085 Timestamped Readout"));

  for (uint8_t sensorIndex = 0; sensorIndex < SENSOR_COUNT; ++sensorIndex) {
    sensorReady[sensorIndex] = initializeSensor(sensorIndex);
    nextSensorRetryUs[sensorIndex] = micros() + SENSOR_RETRY_PERIOD_US;
  }

  Serial.print(F("READY,protocol=FRAME_v1,address=0x"));
  Serial.print(MLX90393_ADDR, HEX);
  Serial.print(F(",sensor_count="));
  Serial.print(SENSOR_COUNT);
  Serial.print(F(",mux_channels=0|2|5|7,rate_hz="));
  Serial.print(SAMPLE_RATE_HZ);
  Serial.println(F(",filter=3,oversampling=1,resolution_bits=16,unit=uT"));

  if (motionReady) {
    Serial.println(F("READY,protocol=MOTION_v1,transport=uart_rvc,rate_hz=50,units=degrees|m_s2"));
  }

  nextFingerSampleUs = micros();
}

void loop() {
  const uint32_t nowUs = micros();
  if (static_cast<int32_t>(nowUs - nextFingerSampleUs) >= 0) {
    const uint32_t sampleStartedUs = nowUs;
    emitFingerFrame(sampleStartedUs);
    nextFingerSampleUs += SAMPLE_PERIOD_US;

    // Avoid a burst of back-to-back samples after an overrun.
    if (static_cast<int32_t>(micros() - nextFingerSampleUs) >= 0) {
      nextFingerSampleUs = micros() + SAMPLE_PERIOD_US;
    }
  }

  // UART-RVC is polled continuously so the BNO stream does not need to block
  // the MLX schedule. The library returns false when no complete packet exists.
  if (motionReady) {
    BNO08x_RVC_Data reading;
    if (rvc.read(&reading)) {
      emitMotionFrame(reading);
    }
  }

  delayMicroseconds(500);
}
