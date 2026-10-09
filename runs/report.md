# Results

| run | stories ppl | CHILDES ppl | Zorro acc | judge ppl (gen) | distinct-2 | rep-3 | trained tok | fwd-only tok | wall (min) |
|---|---|---|---|---|---|---|---|---|---|
| baseline-s0 | 11.30 | 37.70 | 0.650 | 13.04 | 0.369 | 0.031 | 12.00M | 0.00M | 13.8 |
| cradle-no-babble-s0 | 11.39 | 33.94 | 0.659 | 14.13 | 0.342 | 0.041 | 11.55M | 1.35M | 16.4 |
| cradle-no-curriculum-s0 | 11.85 | 38.07 | 0.654 | 15.48 | 0.341 | 0.039 | 11.44M | 1.72M | 15.1 |
| cradle-no-prior-s0 | 12.00 | 34.71 | 0.643 | 13.70 | 0.341 | 0.047 | 11.43M | 1.72M | 15.4 |
| cradle-no-sleep-s0 | 11.29 | 34.53 | 0.654 | 13.33 | 0.367 | 0.041 | 11.88M | 0.37M | 13.9 |
| cradle-s0 | 11.60 | 34.20 | 0.654 | 14.99 | 0.337 | 0.052 | 11.43M | 1.72M | 14.9 |

Stories validation loss (48-window subset) vs. share of compute budget:

| run | 15% | 30% | 60% | 85% | 100% |
|---|---|---|---|---|---|
| baseline-s0 | 3.202 | 2.773 | 2.369 | 2.209 | 2.183 |
| cradle-no-babble-s0 | 4.535 | 2.900 | 2.351 | 2.188 | 2.175 |
| cradle-no-curriculum-s0 | 3.211 | 2.793 | 2.368 | 2.237 | 2.243 |
| cradle-no-prior-s0 | 6.116 | 2.960 | 2.383 | 2.244 | 2.221 |
| cradle-no-sleep-s0 | 4.535 | 2.900 | 2.351 | 2.223 | 2.170 |
| cradle-s0 | 4.535 | 2.900 | 2.351 | 2.223 | 2.197 |

## baseline-s0 samples

- once upon a time, in a big forest, there lived a little bear named benny. benny lived in the forest to go on an adventure. he would play with all his friends, and he would wander around the forest. one day, benny saw a big, beautiful tree that he really wanted to touch it. he tried to twist it, but it was too heavy for him to reach. then, he heard something he had never seen before. it was a bush! benny was scared and he didn't know what to say. suddenly
- once upon a time, in a small yard, there was a little girl named lily. one day, she wanted a haircut. lily wanted to help her friend, her friend, max. she asked her mom if she could help her with her toy. her mom said yes and so lily went to find her friend, max. but when they got to the store, lily saw her big, dirty toy and didn't want to get it. she asked her mom if she had a toy, but her mom said no. lily was sad
- once upon a time, there was a thoughtful girl named sue. sue loved to play outside in the sun. one day, she went outside to play with her friends. they walked and watched the sun together. sue and her friends ran around and went on the night. they played in the sun and soon, until it was time to go home. sue and her friends spent their day with the sun and it was time to go home. at the morning, sue had so much fun with her family. she told her mom about the sun

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
