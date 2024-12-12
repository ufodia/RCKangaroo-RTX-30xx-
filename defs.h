// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once 

#pragma warning(disable : 4996)

typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef int i32;
typedef unsigned short u16;
typedef short i16;
typedef unsigned char u8;
typedef char i8;


#define MAX_GPU_CNT			32
#define BLOCK_SIZE			256
#define PNT_GROUP_CNT		24

#define LDS_SIZE_A			(68 * 1024)
#define LDS_SIZE_B			(98 * 1024)
#define LDS_SIZE_C			(96 * 1024)

// kang type
#define TAME				0  // Tame kangs
#define WILD1				1  // Wild kangs1 
#define WILD2				2  // Wild kangs2

#define GPU_DP_SIZE			48
#define MAX_DP_CNT			(256 * 1024)

//same for all tables
#define JMP_CNT				1024
#define JMP_MASK			(JMP_CNT-1)

#define DPTABLE_MAX_CNT		64

#define MAX_CNT_LIST		(512 * 1024)

#define DP_FLAG				0x8000
#define INV_FLAG			0x4000
#define JMP2_FLAG			0x2000

#define STEP_CNT			2048

#define MD_LEN				8

//#define DEBUG_MODE

//gpu kernel parameters
struct TKparams
{
	u64* Kangs;
	u32 KangCnt;
	u32 BlockCnt;
	u64* L2;
	u64 DP;
	u32* DPs_out;
	u64* Jumps1; //x(32b), y(32b), d(32b)
	u64* Jumps2; //x(32b), y(32b), d(32b)
	u64* Jumps3; //x(32b), y(32b), d(32b)
	u64* JumpsList; //list of all performed jumps, grouped by warp(32) every 8 groups (from PNT_GROUP_CNT). Each jump is 2 bytes: 10bit jump index + flags: INV_FLAG, DP_FLAG, JMP2_FLAG
	u32* DPTable;
	u32* L1S2;
	u64* LastPnts;
	u64* LoopTable;
	u32* dbg_buf;
	u32* LoopedKangs;
};

