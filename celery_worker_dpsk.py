import time  
import os 
class cal_alarm_schedule_definition():
  def __init__(self,
               tag_json_list:list,
               output_from_VLM:str , 
               zone:str,
               when_alarm_neede:str,
               time_range_):
    import datetime
    from zoneinfo import ZoneInfo
    self.listtagjson = tag_json_list
    self.output_from_VLM = output_from_VLM
    today_s_date = datetime.datetime.now(ZoneInfo(zone)).strftime("%Y-%m-%d") 
    self.today_s_date = today_s_date
    self.time_range_ = time_range_
    self.when_alarm_neede = when_alarm_neede
    self.checks ='each'  


def deepseek_load():

  #!pip install transformers accelerate optimum-quanto decord --quiet

  # Deepseek  to event database
  import locale
  locale.getpreferredencoding = lambda: "UTF-8"
  import os
  from huggingface_hub import login 
  import yaml
  if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    with open('hfk.yaml') as f:
      os.environ["HUGGINGFACEHUB_API_TOKEN"] =  yaml.safe_load(f)
    HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
    login(HUGGINGFACEHUB_API_TOKEN)  

  if not os.getenv("DEEPSEEK_API_KEY"):
      with open('dpsk.yaml') as f:
        os.environ["DEEPSEEK_API_KEY"] =  yaml.safe_load(f) 

  return

def getcds(temperature=0,):
  from langchain_deepseek import ChatDeepSeek
  llm  = ChatDeepSeek(
    model="deepseek-chat",
    temperature=temperature,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
  return llm


def get_chain_message(system_message:str,
               human_message:str)-> list:
  messages= [
    SystemMessage(
        content=system_message
    ),
    HumanMessage(
        content=human_message
    )
]

  return messages
def prompt_Wdescrp_Wprofiles_DEEPSEEK(cal_alarm_schedule_ , 
                             ):
  time_range_ = cal_alarm_schedule_.time_range_
  today_s_date = cal_alarm_schedule_.today_s_date
  output_from_VLM = cal_alarm_schedule_.output_from_VLM
  tag_json_list = cal_alarm_schedule_.listtagjson 
  when_alarm_neede = cal_alarm_schedule_.when_alarm_neede   

  prompt = f"""Return only the calendar .ics file of the schedule for the event/period/routine/activity/ies, and no comments, nothing else. 
          """
  system_message=f"""You are an AI assistant writing a schedule for a given event/period/routine/activity/timetable.

            A. Inputs:
            1. Schedule in a given document: {output_from_VLM}.
            2. Information corresponding to a specific person: {str(tag_json_list)}
            
            B. Reference time:
            1. Today is {today_s_date}.  
            
            C. Output format: us the .ics calendar format.  
                1. Person information(if applicable): 
                    BEGIN:VCARD
                    FN:John Doe (Full Name)
                    N:Doe;John;;; (Name components)
                    EMAIL:john.doe@example.com (Email Address)
                    TEL;TYPE=WORK:+1 555 1234 (Phone Number)
                    END:VCARD
                    
                2. Event example with recurring rule and Reminder/alarm information:
                    i.  BEGIN:VEVENT
                        DTSTAMP:20230101T120000
                        DTSTART:20230101T140000
                        DTEND:20230101T150000
                        ORGANIZER:mailto:organizer@example.com
                        ATTENDEE;CN=John Doe;RSVP=TRUE:mailto:attendee1@example.com
                        ATTENDEE;CN=Jane Smith;RSVP=TRUE:mailto:attendee2@example.com
                        SUMMARY:Meeting with John
                        RRULE:FREQ=WEEKLY;BYDAY=WE;UNTIL=20251031T235959
                        BEGIN:VALARM
                        ACTION:DISPLAY
                        TRIGGER;VALUE=DATE-TIME:20230101T135500
                        DESCRIPTION:Reminder: Meeting with John in 5 minutes.
                        END:VALARM
                        END:VEVENT
                    ii. Explanation for alarm:
                        a. ACTION: Specifies the action to be triggered when the alarm occurs. Common values include: 
                            1. DISPLAY: Displays a message to the user. 
                            2. AUDIO: Plays an audio alert. 
                            3. EMAIL: Sends an email. 
                        b. TRIGGER: Defines when the alarm should be triggered.  This should only be a timedelta (duration) to alert before a specific time. 
                        c. DURATION: Specifies the duration of the alarm, especially useful for recurring alarms. 
                        d. REPEAT: An integer that specifies how many times the alarm should repeat.                 
                    
            D. Notes:
            1. if the schedule is a timetable(i.e., the activities given different periods of day for days of a week), it will recur weekly till a given time range end date {time_range_}
            2. If it is a one-off event, the periods will run only for a day or specific set of dates.
            3. Only allow dates that can exist. For example 29 February only can exist in leap years   
            4. Set alarms for {when_alarm_neede}  before the various activities/events.
            5. Note also that some timetables are for Sunday to Thursdays and not Mondays to Fridays. 
            6. Do not put a "Z" at the end of the formatted date times
            7. If no time for an event is given, give the DTSTART as starting at 8am and DTEND as ending at 4pm.
            8. Use RRULE for recurring events/timetables that occur weekly and do not input rrule if they are one-off. Only recur the timetable once at the start of the period and do not have extra events that recur giving the duplicate events.
            
            E. Ensure that the entire .ics file begins with "BEGIN:VCALENDAR" and ends with "END:VCALENDAR", dropping events that cannot be fit within the token limit. 

            """ #consider putting 'transcript string indices'
  human_message=f"""{prompt}"""
  messages_extract_topicstatements = get_chain_message(system_message,human_message)
  prompttemplate = ChatPromptTemplate.from_messages(messages_extract_topicstatements )
  llm_ls=getcds( )
  chain =  (prompttemplate
                    |
                  llm_ls  )
  response = chain.invoke({'prompt':prompt    })
  return response.content.strip()
    
def long_running_task(defined_class ):
    """Simulates a long-running task.""" 
      
    return text_gen1
if __name__=="__main__":
    import os
    import time 
    mainfilepath = "/fastapi/uf/"
    model1, processor1, DEVICE1 = load_qwen_VLM_model() 
    while 1:
        nowfilepaths =  os.listdir(mainfilepath )
        full_paths = []
        nowfilepaths1 = [x for x in nowfilepaths if '.txt' not in x]
        try:
            item = nowfilepaths1[0] 
            
            with open(  os.path.join(mainfilepath,item+'.txt') ,'r') as ff:
                prompt1 = ff.read()
            image_path =   os.path.join(mainfilepath,item  )
            text_gen1 = long_running_task(prompt1,  image_path  )
            with open(  os.path.join(mainfilepath,item+'text_gen.txt'),'w') as ff:
                ff.write(text_gen1)
            os.system('rm '+mainfilepath+item+'.txt')
            os.system('rm '+mainfilepath+item )
            
            time.sleep(0.1)
         
        except Exception :
            
            time.sleep(1)
         
        
        
