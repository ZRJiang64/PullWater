 //單純定義pin輸出，以及輸出的速度
int motorPin_A = 7; //PIL+
int motorPin_AP = 8; //DIR+
int delayTime = 5000; 
int a=1;  
void setup()  
{
  Serial.begin(9600);
  pinMode(motorPin_A, OUTPUT);
  pinMode(motorPin_AP, OUTPUT);
}

void loop()
{   
  if (a==1)
  { 
    digitalWrite(motorPin_AP,HIGH);
    for (int i=0;i<400;i++)
      {
      digitalWrite(motorPin_A, HIGH);
      delayMicroseconds(delayTime);
      digitalWrite(motorPin_A, LOW);
      delayMicroseconds(delayTime);
      }
    
    digitalWrite(motorPin_AP,LOW);
    delay(5000);
    int g=900;
    for (int i=0;i<=400;i++)
      {
      digitalWrite(motorPin_A, HIGH);
      delayMicroseconds(delayTime);
      digitalWrite(motorPin_A, LOW);
      delayMicroseconds(delayTime);
      }
    delay(500);
    digitalWrite(motorPin_A,LOW);
    for (int i=0;i<=0;i++)
      {
      digitalWrite(motorPin_A, HIGH);
      delay(delayTime);
      digitalWrite(motorPin_A, LOW);
      delay(delayTime);
      }
   
   
  a=2;
  Serial.println("stop");
  }

}
