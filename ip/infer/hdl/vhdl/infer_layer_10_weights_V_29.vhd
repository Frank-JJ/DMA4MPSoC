-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity infer_layer_10_weights_V_29_rom is 
    generic(
             DWIDTH     : integer := 15; 
             AWIDTH     : integer := 5; 
             MEM_SIZE    : integer := 32
    ); 
    port (
          addr0      : in std_logic_vector(AWIDTH-1 downto 0); 
          ce0       : in std_logic; 
          q0         : out std_logic_vector(DWIDTH-1 downto 0);
          clk       : in std_logic
    ); 
end entity; 


architecture rtl of infer_layer_10_weights_V_29_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "100110111011110", 1 => "110011111001111", 2 => "001000011101000", 
    3 => "011011010100010", 4 => "001101100011011", 5 => "001100010010101", 
    6 => "111011011111011", 7 => "010011111000000", 8 => "001000110000010", 
    9 => "010000110111000", 10 => "001110011000101", 11 => "011010001101111", 
    12 => "110011000001011", 13 => "111000100111111", 14 => "100101101111001", 
    15 => "100010000110011", 16 => "100010000001100", 17 => "001100011001110", 
    18 => "111010001110110", 19 => "000111000011100", 20 => "011110010011110", 
    21 => "100010010001100", 22 => "000110010111000", 23 => "001010100001100", 
    24 => "010100100011011", 25 => "110010100010010", 26 => "000110111111110", 
    27 => "010110111001000", 28 => "110010010010110", 29 => "101100000110001", 
    30 => "010100010101011", 31 => "001011011101010" );


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

entity infer_layer_10_weights_V_29 is
    generic (
        DataWidth : INTEGER := 15;
        AddressRange : INTEGER := 32;
        AddressWidth : INTEGER := 5);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of infer_layer_10_weights_V_29 is
    component infer_layer_10_weights_V_29_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    infer_layer_10_weights_V_29_rom_U :  component infer_layer_10_weights_V_29_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


