#include<Servo.h>
Servo X ;
Servo Y ;

void setup() {
  
  X.attach(7);
  Y.attach(8);
  Serial.begin(9600);
  X.write(90);
  Y.write(180);
  pinMode(13,OUTPUT);
}
void loop() {
if(Serial.available())
{
 char x =Serial.read();
 int v =Serial.parseInt();
 
  if(x =='a')
  {
    X.write(v);
  }
  if(x =='b')
  {
    Y.write(v);
  }
  if(x=='c')
    {
      tone(13,220,125);
      delay(125);
      tone(13,2093,250);
      delay(250);
      tone(13,82,125);
      delay(125);
    }
  }
}
