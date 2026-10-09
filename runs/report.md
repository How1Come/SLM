# Results

| run | stories ppl | CHILDES ppl | Zorro acc | judge ppl (gen) | distinct-2 | rep-3 | trained tok | fwd-only tok | wall (min) |
|---|---|---|---|---|---|---|---|---|---|
| baseline-plus-prior-s0 | 11.20 | 36.92 | 0.649 | 13.47 | 0.353 | 0.039 | 12.00M | 0.00M | 13.7 |
| baseline-s0 | 11.30 | 37.70 | 0.650 | 13.04 | 0.369 | 0.031 | 12.00M | 0.00M | 13.8 |
| baseline-s1 | 11.34 | 37.91 | 0.648 | 15.99 | 0.379 | 0.036 | 12.00M | 0.00M | 12.7 |
| cradle-listen-only-s0 | 11.08 | 34.27 | 0.655 | 14.46 | 0.370 | 0.042 | 12.01M | 0.00M | 13.4 |
| cradle-listen-only-s1 | 11.10 | 34.26 | 0.642 | 14.77 | 0.372 | 0.043 | 12.01M | 0.00M | 13.3 |
| cradle-no-babble-s0 | 11.39 | 33.94 | 0.659 | 14.13 | 0.342 | 0.041 | 11.55M | 1.35M | 16.4 |
| cradle-no-curriculum-s0 | 11.85 | 38.07 | 0.654 | 15.48 | 0.341 | 0.039 | 11.44M | 1.72M | 15.1 |
| cradle-no-prior-s0 | 12.00 | 34.71 | 0.643 | 13.70 | 0.341 | 0.047 | 11.43M | 1.72M | 15.4 |
| cradle-no-sleep-s0 | 11.29 | 34.53 | 0.654 | 13.33 | 0.367 | 0.041 | 11.88M | 0.37M | 13.9 |
| cradle-s0 | 11.60 | 34.20 | 0.654 | 14.99 | 0.337 | 0.052 | 11.43M | 1.72M | 14.9 |
| cradle-v2-no-babble-s0 | 10.83 | 43.82 | 0.627 | 13.09 | 0.368 | 0.043 | 12.00M | 0.00M | 13.5 |
| cradle-v2-no-babble-s1 | 10.90 | 44.06 | 0.660 | 12.59 | 0.358 | 0.036 | 12.00M | 0.00M | 13.0 |
| cradle-v2-s0 | 11.04 | 43.57 | 0.611 | 13.96 | 0.372 | 0.036 | 11.88M | 0.37M | 14.9 |

Stories validation loss (48-window subset) vs. share of compute budget:

| run | 15% | 30% | 60% | 85% | 100% |
|---|---|---|---|---|---|
| baseline-plus-prior-s0 | 3.211 | 2.793 | 2.368 | 2.210 | 2.184 |
| baseline-s0 | 3.202 | 2.773 | 2.369 | 2.209 | 2.183 |
| baseline-s1 | 3.241 | 2.789 | 2.369 | 2.218 | 2.186 |
| cradle-listen-only-s0 | 4.535 | 2.900 | 2.351 | 2.188 | 2.163 |
| cradle-listen-only-s1 | 4.603 | 2.891 | 2.349 | 2.182 | 2.153 |
| cradle-no-babble-s0 | 4.535 | 2.900 | 2.351 | 2.188 | 2.175 |
| cradle-no-curriculum-s0 | 3.211 | 2.793 | 2.368 | 2.237 | 2.243 |
| cradle-no-prior-s0 | 6.116 | 2.960 | 2.383 | 2.244 | 2.221 |
| cradle-no-sleep-s0 | 4.535 | 2.900 | 2.351 | 2.223 | 2.170 |
| cradle-s0 | 4.535 | 2.900 | 2.351 | 2.223 | 2.197 |
| cradle-v2-no-babble-s0 | 3.280 | 2.743 | 2.295 | 2.156 | 2.122 |
| cradle-v2-no-babble-s1 | 3.271 | 2.735 | 2.312 | 2.164 | 2.138 |
| cradle-v2-s0 | 3.280 | 2.743 | 2.295 | 2.185 | 2.143 |

## baseline-plus-prior-s0 samples

- once upon a time, in a big forest, there lived a little girl named lily. she loved to play in the forest, especially the flowers. one day, she went to the park and found a big, yellow tree. she wanted to have a purple, so she asked her mom, "can i have a diamond?" her mom said, "no, lily, you need a key for your homework! you can fix it." lily was nervous, but she knew her mom was right. she thought about how the key and
- once upon a time, in a small yard, there was a little girl named lily. one day, she wanted a haircut. lily wanted to eat the dinner, but her mom told her to be careful. lily was sad because she didn't want to eat the haircut. later that day, lily's mom gave her a warm bath. lily was happy to help and put the barber in the dishwasher to eat. but then, lily saw the smelly haircut and had a little cry. she didn't notice the bad haircut and
- once upon a time, there was a thoughtful girl named sue. sue loved to play outside in the sun. one day, she went outside to play with her friends. they had so much fun to play in the sun. it was very cool and warm and had lots of fun. sue and her friends played for a long time all day, but it was time to go home. sue went home to tell her friends all about the fun she had to do the same to play. the end.

## baseline-s0 samples

- once upon a time, in a big forest, there lived a little bear named benny. benny lived in the forest to go on an adventure. he would play with all his friends, and he would wander around the forest. one day, benny saw a big, beautiful tree that he really wanted to touch it. he tried to twist it, but it was too heavy for him to reach. then, he heard something he had never seen before. it was a bush! benny was scared and he didn't know what to say. suddenly
- once upon a time, in a small yard, there was a little girl named lily. one day, she wanted a haircut. lily wanted to help her friend, her friend, max. she asked her mom if she could help her with her toy. her mom said yes and so lily went to find her friend, max. but when they got to the store, lily saw her big, dirty toy and didn't want to get it. she asked her mom if she had a toy, but her mom said no. lily was sad
- once upon a time, there was a thoughtful girl named sue. sue loved to play outside in the sun. one day, she went outside to play with her friends. they walked and watched the sun together. sue and her friends ran around and went on the night. they played in the sun and soon, until it was time to go home. sue and her friends spent their day with the sun and it was time to go home. at the morning, sue had so much fun with her family. she told her mom about the sun

## baseline-s1 samples

- once upon a time, in a big forest, there lived a little bear named tom. tom was very happy and loved to play outside. one day, tom wanted to race his way home. he said to tom, "let's go!" tom said, "no, please!" his mom said, "ok, we can play together!" tom was sad and they didn't want the bear to go. they went home and told their mom for it. when they came back home, they saw that the bear was gone and looked very
- once upon a time, in a small yard, there was a little girl named lily. one day, she wanted a haircut. an umbrella was very happy and took her apart. lily would give it to the bird and they played together all day long. the bird would fly high in the sky and find a little girl named lily. "look, tom, i found a yummy cookie!" lily said. "that is my favorite teddy bear," tom replied. "but you don't know how to fix it." lily smiled and said,
- once upon a time, there was a thoughtful girl named sue. sue was very excited to explore the world around her. she loved to explore and discover new things. one day, sue found a big box in the corner. it was very fragile and had many toys. sue picked out many toys and put them in one drawer. sue thought, "maybe we look under it, but stay still?" the toys were sad and started to cry. sue's mom came to see what was wrong. she told sue, "that was my favorite

## cradle-listen-only-s0 samples

- once upon a time, in a big forest, there lived a little bear named benny. benny loved to play in the forest, especially the forest. one day, benny saw a big tree with many leaves. he wanted to climb, but his mom said it was too small for benny. benny was frightened, as he could, but he listened to his mom. benny felt safe and scared. then, he had an idea. he thought of a good idea. he ran outside and tried to climb up the tree. but the tree was
- once upon a time, in a small yard, there was a little girl named lily. one day, she wanted a haircut. lily wanted to have a haircut, but her mommy said she wanted to know how. the barber was very kind, so she walked away in a small house. soon, the day came over and lily came over. her mommy took her to the hospital. the doctor and lily were very happy to see the doctor and they all loved how they would do anything to fix the barber. the doctor and lily continued to
- once upon a time, there was a thoughtful girl named sue. sue was very excited to explore the room. the sun was so big and bright that sue had lots of fun. at home, the sun was setting dark shining brightly. sue went to the park to play, but as soon as she opened the door, she saw a big, scary dog! sue was scared and started to cry. but then, sue started to cry and her friend came over to tom. "don't worry, sue. we can fix it!"

## cradle-listen-only-s1 samples

- once upon a time, in a big forest, there lived a little bear named tommy. tommy was very happy because he loved to eat his fruit. they would eat pastels every day and make their fruit. one day, tommy found a red fruit on the ground and wanted to eat their fruit. "what's this?" asked tommy. "my name is tommy, tommy. the salad is too yummy!" tommy said. he thought for a moment and said, "yes, please don't eat it." tommy and tommy looked around
- once upon a time, in a small yard, there was a little girl named lily. one day, she wanted a haircut. lily wanted to take the haircut, so she asked her mom if she liked the haircut. her mom said, "let's make a haircut. it's not very comfortable." lily was sad and cried. "don't worry, lily," her mom said. "but it can't be my haircut. it must make us feel better." but lily didn't listen. she didn't notice the haircut cry.
- once upon a time, there was a thoughtful girl named sue. sue loved to play outside in the sun. one day, sue went outside to play with her friends. they had so much fun that together. sue and her friends wanted to play in the mud. sue said, "yes, i am going to rain. can i have my rain? i will use it for you." the rain was coming from the rain and sue said, "those are wet and wet. the mud is dry." sue and her friends all loved

## cradle-no-babble-s0 samples

- once upon a time, in a big forest, there lived a little bear named benny. benny loved to play in the forest, especially the forest. one day, benny saw a big tree with many leaves. he wanted to climb up the tree, but he knew it was okay. his friend replied, "what's wrong, benny?" benny replied, "i don't want to pass the tree too. let's go to the tree." benny's mom said, "yes, benny! that's not nice." benny and his friend
- once upon a time, in a small yard, there was a little girl named lily. one day, she wanted a haircut. lily wanted to take the haircut look at her. she asked her mom if she could take a haircut. her mom said, "lily, you are so happy with your new haircut! let's be our friends." her mom took her to the park. when lily got home, she said, "can i play with your haircut?" but she had a little bit of work. it wasn't as important as
- once upon a time, there was a thoughtful girl named sue. sue was very excited to explore the world. she loved going to the top of the trees and explore the world she had never seen before. one day, sue walked up to a big park and saw a big bird. the bird was scared and scared, but sue was determined to find it. as she walked around, she saw a frog coming from the park. sue climbed and slid down the hill until she saw the frog. she knew it was not fun to play with

## cradle-no-curriculum-s0 samples

- once upon a time, in a big forest, there lived a little girl named lily. she loved to play in the forest, especially the flowers. one day, she went to the park and found a big, yellow spot. she wanted to be happy with her family and family. lily was playing with her ball. when she got there, she realized that she had been trying to be punished. so, she went to the park with her toy cars. she felt sad and didn't know what to do. suddenly, she remembered
- once upon a time, in a small yard, there was a little girl named lily. one day, she wanted a haircut. lily wanted to eat the dinner, but her mom told her not to worry. lily was sad because she didn't want to make a haircut. later that day, lily's mom gave lily a warm bath. lily was happy to help and put the barber in her bed. after breakfast, lily and her mom got home. afterwards, she had a little bit of ice cream. she put her toys in
- once upon a time, there was a thoughtful girl named sue. sue loved to play outside in the sun. one day, she went outside to play with her friends. they had so much fun to take a deep breath. it was a cool day for spot to go on an adventure! sue was so happy to have a special adventure, and she started to climb trees and she felt the warm breeze. as she was swinging, she tried to run away, but she couldn't get out for her nap. sue cried and ran as

## cradle-no-prior-s0 samples

- once upon a time, in a big forest, there lived a little bear named fin. in a small town, there was a little boy named timmy. timmy loved to play hide-and-seek with his friends. one day, timmy's mom told him she could ride a safety place to rest. when the sun was shining, timmy's mom said, "we're playing near your nest!" timmy was scared, so they ran to the other house together. as they walked, timmy saw a big, scary dog.
- once upon a time, in a small yard, there was a little girl named lily. one day, she wanted a haircut. lily wanted to eat the haircut, but her mommy said, "no, lily. the haircut is mine!" lily didn't want to give her a lollipop. soon, lily's mommy came outside and came over. her mommy took her to the doctor. lily was very excited and didn't want to eat her snack. she asked her mommy what she had to say, but her mom said no. lily didn
- once upon a time, there was a thoughtful girl named sue. sue loved to play outside in the park with her friends. one day, sue found a key on the ground. she ran to the door and said, "mom, i want you to touch it. what are you doing?" her mom smiled and said, "i am going to be a new place." sue and her friends went to the park. they started to play and have fun. they had so much fun playing with the key until they found a new treasure.

## cradle-no-sleep-s0 samples

- once upon a time, in a big forest, there lived a little bear named tommy. tommy was very curious and loved to play outside. one day, he wanted to explore a big house. he said he would like to go outside and try to take a breath. he walked up his door and saw a bird. the bird was scared and ran away. tommy's mommy came outside and saw what was happening. she saw the bird and walked away from the sky. tommy asked her what was wrong and she told him not to worry.
- once upon a time, in a small yard, there was a little girl named lily. one day, she wanted a haircut. lily wanted to take the haircut look at her. she asked her mom if she could take a haircut. her mom said, "sure, i will ask you if you want to take a haircut first." lily smiled and said, "sure, let's wipe your hands and show it to my haircut!" from that day on, lily always made sure to always put things on her hair to make her haircut.
- once upon a time, there was a thoughtful girl named sue. sue was very careful and always asked her mom if she could go to the park. her mom said, "don't worry, the sun is the most important day." one day, sue and her mom went to the park and saw a little girl playing with a new collar. she said, "i want to play with the collar, but you can play with it on the bench." sue watched as she walked by the park. she saw a big swing and a balloon

## cradle-s0 samples

- once upon a time, in a big forest, there lived a little bear named benny. benny lived in the forest with his friends, but they always had a great time together. benny was a great day and he would always love to swim together. one day, benny and his friends went their walk to the park. when the park was there, it met a big, beautiful bird! the bird said, "hi, benny! i have a toy bird. he is a big red dog, a bird. can you help me carry
- once upon a time, in a small yard, there was a little girl named lily. one day, she wanted a haircut. lily wanted to eat the haircut, but her mommy said she wanted to know how to eat the haircut. lily was sad and didn't know what to do. soon, lily's mommy came back to the park. her mommy took her to the park. lily was very excited and didn't want to stop playing with her. afterwards, she had a great time playing. she forgot to be able to make
- once upon a time, there was a thoughtful girl named sue. sue was very excited to explore the world. she loved going to the top of the world. one day, she went to the park with her mom and dad. they saw a big park and decided to go on the swings and play. sue had a fun time and then, she started to climb. when she reached the park, she saw something moving through there. on the bench, there was a big, heavy wind. she was very scared and cried. suddenly,

## cradle-v2-no-babble-s0 samples

- once upon a time, in a big forest, there lived a little girl named lily. she loved to play outside in her backyard. one day, they found a big leaf. lily was scared, but they wanted to play. she tried to run away, but she couldn't. lily's mom was not happy. lily was nervous, but she loved animals. she decided to play a game near her house. as they went, lily saw a toy cars. she felt sad because she didn't want to sit on them again. her
- once upon a time, in a small yard, there was a little girl named lily. one day, she wanted a haircut. lily asked her mom if she could have her haircut. her mom said yes, and the haircut was very spicy. lily was so happy to have a haircut. soon, lily's mom said, "lily, it's time to clean up the haircut. you can need a haircut." lily was happy and hugged her mom. afterwards, she had a haircut in her hair. it was still soft and shiny
- once upon a time, there was a thoughtful girl named sue. sue loved to play outside in the park. one day, sue found a big, yummy bug on the ground. she asked her friend, sally, and they walked together. sue was happy to have such a fun ride in the park. sue played with her bug all day, and she was very happy when she felt thirsty. she decided to leave the bug outside and find a nap. at the park, sue saw a little bug who was really happy. she knew that

## cradle-v2-no-babble-s1 samples

- once upon a time, in a big forest, there lived a little girl named lily. she loved to play with her friends in the forest. one day, lily's mommy asked her to help make sure they were like to clean up the animals. lily didn't want to do all that, but she said he would make the animals happy. while playing, lily's mommy came outside and saw her too shy. she asked lily, "mommy, can i help you." lily replied, "yes, i can help you." her mommy
- once upon a time, in a small yard, there was a little girl named lily. one day, she wanted a haircut. she asked her mommy if she could have her haircut. her mommy said yes, and the little girl was happy to have her haircut. but when she looked inside, she saw her little girl crying. "mommy, her haircut!" she said, "yes, let's play again!" and it said in their little girl. afterwards, she had a hard time playing. she looked at the haircut and said
- once upon a time, there was a thoughtful girl named sue. sue loved to play with her dolls and draw pictures. one day, sue went to the park with her mom. at the park, sue saw a pretty slide. she wanted to slide down and have fun. she ran to the slide and climbed on the slide. sue sat down and started to climb. she felt proud of herself. she showed her mom how to slide on the slide. her mom had a great time at the park. sue took turns. she pushed the

## cradle-v2-s0 samples

- once upon a time, in a big forest, there lived a little girl named lily. she loved to play in the forest in the forest. one day, lily saw a big, scary dog. she wanted to pet it. she went to the dog and asked her what was wrong. the dog was walking as lily showed the dog her favorite toy. lily wanted to pet the dog, but her dog started to bark. lily felt sad because she missed playing outside. she asked her friend, max, if she could pet the dog.
- once upon a time, in a small yard, there was a little girl named lily. one day, she wanted a haircut. lily asked her mom if she could have her haircut. her mom said yes, and the haircutcage was very much fun because she could not play with it. lily felt happy that she could help but feel scared her knee. as she put the barber in her hands, her mommy said it was time to go home. lily went home and had a great time playing. she forgot to put her haircut in
- once upon a time, there was a thoughtful girl named sue. sue loved going on a trip with her family. one day, sue's mom asked her to go to the park. after breakfast, sue and her family went to the park for bed. while they were there, sue noticed a big, fluffy bag. it had many pictures, but it was very cold. she wanted to go home and rest. so, she went on a nap. at first, sue woke up and she was happy to see him. suddenly, she
