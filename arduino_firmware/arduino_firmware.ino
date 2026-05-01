// Furuta firmware for autonomous RL training (sim-to-real fine-tune).
// Convention: theta_pend = 0 at UPRIGHT, +-pi at HANGING.
// Differences vs. the old firmware:
//   * Removed latching safety cutoff and SAFE_PEND_ANGLE altogether.
//     The pendulum is ALLOWED to swing through the full +-pi range so the
//     agent can learn swing-up on hardware.
//   * Watchdog (motor-off if Python is silent) is kept.

#include <Arduino.h>

const int encArmPinA  = 20;
const int encArmPinB  = 21;
const int encPendPinA = 19;
const int encPendPinB = 18;
const int motArmPWM   = 5;
const int motArmDir   = 4;

const float PEND_SIGN  = -1.0;
const float ARM_SIGN   =  1.0;
const float MOTOR_SIGN =  1.0;

const float SUPPLY_VOLTAGE = 12.0;
const int   ENC_CPR        = 2000;
const float TICK_TO_RAD    = (2.0 * PI) / ENC_CPR;

const int   LOOP_HZ        = 100;
const float DT             = 1.0 / LOOP_HZ;
const int   SAFETY_TIMEOUT = 100;   // ms without command -> motor off

const float DEADZONE = 0.5;
const float THRESH   = 0.5;

struct StatePacket {
  float thetaPend;
  float thetaArm;
  float omegaPend;
  float omegaArm;
};

volatile long countArm  = 0;
volatile long countPend = 0;
long  lastCountArm      = 0;
long  lastCountPend     = 0;
float omegaArmFiltered  = 0.0;
float omegaPendFiltered = 0.0;

volatile bool tickFlag  = false;
unsigned long lastCommandTime = 0;

void isrArmA();
void isrArmB();
void isrPendA();
void isrPendB();
void applyVoltage(float volts);
void stopMotor();

void setupTimer1() {
  noInterrupts();
  TCCR1A = 0; TCCR1B = 0; TCNT1 = 0;
  OCR1A  = 624;
  TCCR1B = (1 << WGM12) | (1 << CS12);
  TIMSK1 = (1 << OCIE1A);
  interrupts();
}
ISR(TIMER1_COMPA_vect) { tickFlag = true; }

void setup() {
  Serial.begin(500000);

  pinMode(encArmPinA,  INPUT_PULLUP);
  pinMode(encArmPinB,  INPUT_PULLUP);
  pinMode(encPendPinA, INPUT_PULLUP);
  pinMode(encPendPinB, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(encArmPinA),  isrArmA,  CHANGE);
  attachInterrupt(digitalPinToInterrupt(encArmPinB),  isrArmB,  CHANGE);
  attachInterrupt(digitalPinToInterrupt(encPendPinA), isrPendA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encPendPinB), isrPendB, CHANGE);

  pinMode(motArmPWM, OUTPUT);
  pinMode(motArmDir, OUTPUT);
  stopMotor();

  setupTimer1();
  lastCommandTime = millis();
}

void loop() {
  if (tickFlag) {
    tickFlag = false;

    noInterrupts();
    long cArm  = countArm;
    long cPend = countPend;
    interrupts();

    long dArm  = cArm  - lastCountArm;
    long dPend = cPend - lastCountPend;

    float rawVelArm  = (dArm  * TICK_TO_RAD) / DT;
    float rawVelPend = (dPend * TICK_TO_RAD) / DT;

    omegaArmFiltered  = 0.25 * rawVelArm  + 0.75 * omegaArmFiltered;
    omegaPendFiltered = 0.25 * rawVelPend + 0.75 * omegaPendFiltered;

    float pendAngle = cPend * TICK_TO_RAD * PEND_SIGN;

    StatePacket pkt;
    pkt.thetaPend = pendAngle;
    pkt.thetaArm  = cArm * TICK_TO_RAD * ARM_SIGN;
    pkt.omegaPend = omegaPendFiltered * PEND_SIGN;
    pkt.omegaArm  = omegaArmFiltered  * ARM_SIGN;

    uint16_t syncWord = 0xABCD;
    Serial.write((byte*)&syncWord, sizeof(syncWord));
    Serial.write((byte*)&pkt, sizeof(pkt));

    lastCountArm  = cArm;
    lastCountPend = cPend;
  }

  if (Serial.available() >= 4) {
    while (Serial.available() >= 8) {
      uint8_t trash[4];
      Serial.readBytes(trash, 4);
    }
    float voltageCommand;
    Serial.readBytes((byte*)&voltageCommand, 4);
    lastCommandTime = millis();
    applyVoltage(voltageCommand);
  }

  if (millis() - lastCommandTime > SAFETY_TIMEOUT) {
    stopMotor();
  }
}

void applyVoltage(float volts) {
  volts *= MOTOR_SIGN;
  if (volts >  THRESH)      volts += DEADZONE;
  else if (volts < -THRESH) volts -= DEADZONE;
  else                      volts  = 0.0;

  volts = constrain(volts, -SUPPLY_VOLTAGE, SUPPLY_VOLTAGE);
  float duty = constrain(volts / SUPPLY_VOLTAGE, -1.0, 1.0);
  int pwm = (int)(fabs(duty) * 255.0 + 0.5);
  digitalWrite(motArmDir, (duty >= 0.0) ? HIGH : LOW);
  analogWrite(motArmPWM, pwm);
}

void stopMotor() { analogWrite(motArmPWM, 0); }

void isrArmA()  { (digitalRead(encArmPinA) != digitalRead(encArmPinB))  ? countArm++  : countArm--;  }
void isrArmB()  { (digitalRead(encArmPinA) == digitalRead(encArmPinB))  ? countArm++  : countArm--;  }
void isrPendA() { (digitalRead(encPendPinA) != digitalRead(encPendPinB)) ? countPend-- : countPend++; }
void isrPendB() { (digitalRead(encPendPinA) == digitalRead(encPendPinB)) ? countPend-- : countPend++; }
