# Failure analysis: raw evidence

## Intent confusions (LLM), top pairs with examples

### 6x gold=DELIVERY_ADDRESS -> pred=ORDER_STATUS

- ex 50 (conf 0.95): I'm expecting a parcel today (bday gift for someone) and courier guy is not answering the calls. Disappointed
- ex 60 (conf 0.9): order a book, when i tried to track it shows delivered to tamanna i don't know who she is not in my contact

### 5x gold=ORDER_STATUS -> pred=SERVICE_COMPLAINT

- ex 14 (conf 0.85): can you stop using your own horrible delivery services? 1 package arrived fine 1 package missing. Complete garbage.
- ex 16 (conf 0.8): is there any one who can help regarding produsct status or any thing no one is taking responsibility

### 4x gold=PAYMENT -> pred=REFUND

- ex 23 (conf 0.9): Haven't got my cashback yet. Hellppp!!!!!
- ex 31 (conf 0.9): Contacted many times through email. Still not received my Cashback for my earlier order. Sort out this.

### 3x gold=ACCOUNT -> pred=OTHER

- ex 54 (conf 0.65): another disappointment...order an e-voucher instead and someone has managed to hack in and steal it within seconds of it being available..
- ex 156 (conf 0.6): recived my last order from Amazon.in now I am going to delete the account in few daya

### 3x gold=ORDER_STATUS -> pred=OTHER

- ex 51 (conf 0.85): how long does it take to get a star wars battlefront 2 beta code for pre ordering??
- ex 105 (conf 0.75): When does a pre order item ship relative to its release date?? ex. If release is Nov.7 when will it ship, I have prime

### 3x gold=OTHER -> pred=SERVICE_COMPLAINT

- ex 36 (conf 0.85): this is now getting ridiculous I have been let down 2 times now and want this sorted out.
- ex 129 (conf 0.9): . You guys, Monroe was probably the best customer service rep I've ever had. Incredibly helpful and threw in a few jokes. Also, the man's got a voice for radio...y'all need to get this guy on the airwaves, ASAP.

### 2x gold=RETURN_CANCEL -> pred=DELIVERY_ADDRESS

- ex 138 (conf 0.9): no one came today as well! Tomorrow is the only off I've, after that I won't b available for return on my address.
- ex 195 (conf 0.7): Hi i have a query i live in Marthahalli bridge area (Bengaluru) . can you tell me where is the nearest location for phone exchange ???

### 2x gold=OTHER -> pred=DELIVERY_ADDRESS

- ex 13 (conf 0.6): I appreciate my product being protected but could a smaller box not be found?
- ex 83 (conf 0.85): Hi, I'm a Prime member with Amazon.de located in Belgium. I want to buy a Fire TV Stick, but the site keeps telling me that they can not ship it to BE. Is there another way to get it shipped to Belgium?

## Missed escalations (gold=human, agent=auto): 28

- ex 18 [REVIEW_CONTENT, conf 0.60]: I preordered amazon echo but want to go for amazon plus now. Anyway to reach out to seller? I want another invitation.
  - reasons: standard flow
- ex 22 [APP_WEBSITE, conf 0.80]: my orders page is redirecting me to the "manage your kindle" page. also got a weird email from __email__?
  - reasons: standard flow
- ex 24 [REFUND, conf 0.70]: I havnt got my refund yet...for my order no. 406-5219321-0534711, one of the red bull can was damaged.
  - reasons: standard flow
- ex 32 [DAMAGED_OR_WRONG, conf 0.85]: Its manufacturing default in that Xperia It's hardware prblm frm manufacturing,still they are not accepting #sonyxperia
  - reasons: standard flow
- ex 39 [OTHER, conf 0.50]: Help me please.
  - reasons: standard flow
- ex 43 [DELIVERY_ADDRESS, conf 0.95]: thanks to ur driver 2day my bf n I lost out on our #snes bc they didn't use the locker but "delivered" them somewhere
  - reasons: standard flow
- ex 52 [DAMAGED_OR_WRONG, conf 0.92]: Initiate exchange for a defective product and received again another defective..what kind of cheating is it
  - reasons: standard flow
- ex 60 [DELIVERY_ADDRESS, conf 0.90]: order a book, when i tried to track it shows delivered to tamanna i don't know who she is not in my contact
  - reasons: standard flow
- ex 66 [DAMAGED_OR_WRONG, conf 0.85]: been waiting since 20th November for my order via prime, after 2 failed deliveries the book finally arrived and is ripped...
  - reasons: standard flow
- ex 96 [DAMAGED_OR_WRONG, conf 0.90]: Hi I bought something from a 3rd party through amazon. Item arrived broken and inadequately packaged. No response to messages.
  - reasons: standard flow
- ex 106 [ORDER_STATUS, conf 0.90]: Dear , Why do I pay to have next day delivery and then the carrier f*&k$ it up (UPS) and it takes longer? Getting tired of wasting my money #AmazonPrime #notdeliveredaspromised
  - reasons: standard flow
- ex 112 [RETURN_CANCEL, conf 0.85]: Hej, I ordered a Nintendo Switch as part of the Cyber Monday deal. I got though and placed an order but you cancelled it as you had a problem with my card. Do I have any rights to re-order or do I jus
  - reasons: standard flow

## False alarms (gold=auto, agent=human): 40

- ex 0 [DELIVERY_ADDRESS]: Amagent refusing delivery packets in Chaura Raghunath our village sector 22 noida, reason no House no for village. Now having new rule .. lol
  - reasons: non-English message: out of scope for the English-only agent
- ex 6 [OTHER]: Hiya, another Kids Edition HD8 question, how can I get music on there for the kids on THEIR profile???!
  - reasons: non-English message: out of scope for the English-only agent
- ex 14 [ORDER_STATUS]: can you stop using your own horrible delivery services? 1 package arrived fine 1 package missing. Complete garbage.
  - reasons: non-English message: out of scope for the English-only agent; high-risk intent 'SERVICE_COMPLAINT' (money/account/support-complaint lanes)
- ex 15 [OTHER]: Why there is moto c plus in mobile exchange list
  - reasons: non-English message: out of scope for the English-only agent
- ex 19 [ACCOUNT]: My subscription of amazon prime is going to end soon, so i thought not to renew it As the services r not good lately
  - reasons: high-risk intent 'SERVICE_COMPLAINT' (money/account/support-complaint lanes)
- ex 20 [DELIVERY_ADDRESS]: i try to order on certain days so i can be off from work to receive the package because my rental office wont take them anymore,amazon is really screwing up lately and it makes me 
  - reasons: lexicon trigger 'churn_threat'
- ex 23 [PAYMENT]: Haven't got my cashback yet. Hellppp!!!!!
  - reasons: non-English message: out of scope for the English-only agent; high-risk intent 'REFUND' (money/account/support-complaint lanes)
- ex 31 [PAYMENT]: Contacted many times through email. Still not received my Cashback for my earlier order. Sort out this.
  - reasons: high-risk intent 'REFUND' (money/account/support-complaint lanes)

## Worst 8 agent drafts by judge score

- ex 174 (score 2.5): Seriously guys!? #Amazon
  - draft: Hello! We're sorry to hear you're having a frustrating experience. Please send us a direct message or visit our support center at <link> so we can assist you privately. ^AM
- ex 15 (score 4.0): Why there is moto c plus in mobile exchange list
  - draft: For the most accurate information regarding trade-in eligibility, please visit our secure support site here: <link>. We're happy to help guide you through the process! ^JS
- ex 128 (score 4.0): Hi. I have issues regarding exchange of mobile fone. can your agent reach out to me on call?
  - draft: We can definitely help you with your exchange issue! Please send us a direct message with more details so we can assist you securely. ^AJ
- ex 69 (score 4.0): I have raised a return request 408-7456269-9685902 can you check is it refund or replacement?
  - draft: For checking the status of your return, please visit our secure support center at <link>. We're happy to help with any other questions there! ^AG
- ex 135 (score 4.0): , I have Prime and deployed to Poland. I get the service area restriction message. How can I get Prime to work here?
  - draft: We're sorry you're running into service restrictions! Please send us a direct message with more details so we can look into this for you. ^JD
- ex 38 (score 4.0): is amzon pantry available to satna mp 485001,pls tell
  - draft: We can check availability for you securely! Please visit our help center here: <link> to get the best assistance. ^JK
- ex 122 (score 4.0): . tried many times to call but no no chance ... What a pathetic service guys ... U won't last for long if u will be like his in India .taking money and not deli
  - draft: We sincerely apologize for the difficulty reaching us and the delivery issues you've experienced. Please send us a direct message or visit our support page here: <link> ^AM
- ex 195 (score 4.0): Hi i have a query i live in Marthahalli bridge area (Bengaluru) . can you tell me where is the nearest location for phone exchange ???
  - draft: We're sorry, but we don't handle local service locations. Please send us a direct message with more details so we can point you in the right direction. ^AM

## Best 3 agent drafts by judge score

- ex 78 (score 5.0): My Amazon shopping app offered me an 30 day free trial for prime but when I go to Prime Membership it says "Your membership will be active s
  - draft: We understand this is confusing! Please check your membership status securely by visiting this link: <link>. ^AG
- ex 177 (score 5.0): . will you ever get a delivery right? Yet again delivery instructions have being ignored. Already contacted the head of logistics
  - draft: We're so sorry to hear about the delivery issues you're experiencing. Please send us a direct message so we can look into this for you privately. ^JM
- ex 152 (score 5.0): Amazon Prime is a JOKE! Two-day shipping is apparently FIVE-day shipping, DONT WASTE YOUR MONEY ON THEIR LIES
  - draft: I'm so sorry to hear about your shipping experience! Please send us a direct message with your order details so we can look into this for you. ^AM

## Safety scan of agent drafts (regex heuristics)

- none flagged
