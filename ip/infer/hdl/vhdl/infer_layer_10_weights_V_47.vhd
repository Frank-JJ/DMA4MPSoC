-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2020.2 (64-bit)
-- Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity infer_layer_10_weights_V_47_rom is 
    generic(
             DWIDTH     : integer := 16; 
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


architecture rtl of infer_layer_10_weights_V_47_rom is 

signal addr0_tmp : std_logic_vector(AWIDTH-1 downto 0); 
type mem_array is array (0 to MEM_SIZE-1) of std_logic_vector (DWIDTH-1 downto 0); 
signal mem : mem_array := (
    0 => "1011111101000101", 1 => "1110000100010001", 2 => "1100111011111101", 
    3 => "1101110011111011", 4 => "1101110110100011", 5 => "1101111010010000", 
    6 => "1101000100111101", 7 => "1101100110000111", 8 => "1111100111100011", 
    9 => "0000100011010011", 10 => "1100110110001011", 11 => "1100110110001001", 
    12 => "0000011101111000", 13 => "1101011001100110", 14 => "0000110011100011", 
    15 => "1101011100010011", 16 => "1110100110110110", 17 => "0001010011110010", 
    18 => "1110011011011010", 19 => "0010011111001010", 20 => "0000110010101001", 
    21 => "1100111111100000", 22 => "1111010011010111", 23 => "0000010010001000", 
    24 => "1100111010100010", 25 => "1100010100011100", 26 => "0011010110101111", 
    27 => "0001110000011000", 28 => "0000111000101111", 29 => "0010100111001000", 
    30 => "0010001011110010", 31 => "1110010101100111" );


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

entity infer_layer_10_weights_V_47 is
    generic (
        DataWidth : INTEGER := 16;
        AddressRange : INTEGER := 32;
        AddressWidth : INTEGER := 5);
    port (
        reset : IN STD_LOGIC;
        clk : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR(AddressWidth - 1 DOWNTO 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR(DataWidth - 1 DOWNTO 0));
end entity;

architecture arch of infer_layer_10_weights_V_47 is
    component infer_layer_10_weights_V_47_rom is
        port (
            clk : IN STD_LOGIC;
            addr0 : IN STD_LOGIC_VECTOR;
            ce0 : IN STD_LOGIC;
            q0 : OUT STD_LOGIC_VECTOR);
    end component;



begin
    infer_layer_10_weights_V_47_rom_U :  component infer_layer_10_weights_V_47_rom
    port map (
        clk => clk,
        addr0 => address0,
        ce0 => ce0,
        q0 => q0);

end architecture;


