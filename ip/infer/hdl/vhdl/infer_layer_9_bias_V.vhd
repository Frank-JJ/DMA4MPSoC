-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity infer_layer_9_bias_V_rom is 
    generic(
             DWIDTH     : integer := 14; 
             AWIDTH     : integer := 6; 
             MEM_SIZE    : integer := 64
    ); 
    port (
          addr0      : in std_logic_vector(AWIDTH-1 downto 0); 
          ce0       : in std_logic; 
          q0         : out std_logic_vector(DWIDTH-1 downto 0);
          clk       : in std_logic
    ); 
end entity; 


architecture rtl of infer_layer_9_bias_V_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "01010001111000", 1 => "11111010111000", 2 => "11011000001010", 
    3 => "11111100010110", 4 => "11110010101010", 5 => "11111001110110", 
    6 => "11010000111011", 7 => "11110111101100", 8 => "11011001010100", 
    9 => "11110101001101", 10 => "11110100001111", 11 => "11111000111000", 
    12 => "11110111010010", 13 => "11010010100101", 14 => "01010110010000", 
    15 => "11110010010011", 16 => "11011110001010", 17 => "11110111001110", 
    18 => "11110001111101", 19 => "11111001000000", 20 => "00000000000000", 
    21 => "11010010111000", 22 => "11110110000100", 23 => "11010011010110", 
    24 => "01100000101011", 25 => "01011010001011", 26 => "11110100100010", 
    27 => "11110011011011", 28 => "11110111100001", 29 => "11110101010000", 
    30 => "11111010011111", 31 => "00111110101001", 32 => "11100100101000", 
    33 => "11110100110101", 34 => "11110011010111", 35 => "11011010011100", 
    36 => "11101101100110", 37 => "11010100010110", 38 => "01000000000111", 
    39 => "11111000001001", 40 => "00001000010110", 41 => "11111111010011", 
    42 => "11110001110100", 43 => "11110011100110", 44 => "11110000111011", 
    45 => "11111011100000", 46 => "00111001011001", 47 => "11110110001001", 
    48 => "11111111000111", 49 => "00000000000000", 50 => "01001001000000", 
    51 => "11011101101000", 52 => "11111011100011", 53 => "11111010111000", 
    54 => "00111000101101", 55 => "11110010101001", 56 => "00000000000000", 
    57 => "01000100011011", 58 => "00010010000100", 59 => "11010111100110", 
    60 => "11111001000010", 61 => "11100110111001", 62 => "11110000100100", 
    63 => "11100110010110" );


begin 


memory_access_guard_0: process (addr0) 
begin
      addr0_tmp <= addr0;
--synthesis translate_off
      if (CONV_INTEGER(addr0) > mem_size-1) then
           addr0_tmp <= (others => '0');
      else 
           addr0_tmp <= addr0;
      end if;
--synthesis translate_on
end process;

p_rom_access: process (clk)  
begin 
    if (clk'event and clk = '1') then
        if (ce0 = '1') then 
            q0 <= mem(CONV_INTEGER(addr0_tmp)); 
        end if;
    end if;
end process;

end rtl;

Library IEEE;
use IEEE.std_logic_1164.all;

entity infer_layer_9_bias_V is
    generic (
        DataWidth : INTEGER := 14;
        AddressRange : INTEGER := 64;
        AddressWidth : INTEGER := 6);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of infer_layer_9_bias_V is
    component infer_layer_9_bias_V_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    infer_layer_9_bias_V_rom_U :  component infer_layer_9_bias_V_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


