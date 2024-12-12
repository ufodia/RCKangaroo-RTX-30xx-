(c) 2024, RetiredCoder (RC)

RCKangaroo is free and open-source (GPLv3).
This software demonstrates efficient GPU implementation of SOTA Kangaroo method for solving ECDLP. 
It's part #3 of my research, you can find more details here: https://github.com/RetiredC

Features:

- Lowest K=1.15, it means 1.8 times less required operations compared to classic method with K=2.1, also it means that you need 1.8 times less memory to store DPs.
- Fast, about 8GKeys/s on RTX 4090.
- Keeps DP overhead as small as possible.
- Supports ranges up to 170 bits.
- Both Windows and Linux are supported.

Limitations:

- Only RTX 40xx and newer GPUs are supported.
- No advanced features like networking, saving/loading DPs, etc.

Command line parameters:

<b>-gpu</b>		which GPUs are used, for example, "035" means that GPUs #0, #3 and #5 are used. If not specified, all available GPUs are used. 

-pubkey		public key to solve, both compressed and uncompressed keys are supported. If not specified, software starts in benchmark mode and solves random keys. 

-start		start offset of the key, in hex. Mandatory if "-pubkey" option is specified. For example, for puzzle #85 start offset is "1000000000000000000000". 

-range		bit range of private the key. Mandatory if "-pubkey" option is specified. For example, for puzzle #85 bit range is "84" (84 bits). Must be in range 32...170. 

-dp		DP bits. Must be in range 14...60. Low DP bits values cause larger DB but reduces DP overhead and vice versa. 

When public key is solved, software displays it and also writes it to "RESULTS.TXT" file. 

Sample command line for puzzle #85:

RCKangaroo.exe -dp 16 -range 84 -start 1000000000000000000000 -pubkey 0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a


