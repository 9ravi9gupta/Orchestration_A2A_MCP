from mcp.server.fastmcp import FastMCP

mcp=FastMCP("Math")

@mcp.tool()
def add(a:int,b:int)->int:
    """_summary_
    Add to numbers
    """
    return a+b

@mcp.tool()
def multiply(a:int,b:int)-> int:
    """Multiply two numbers"""
    return a*b

@mcp.tool()
def weather(state:str) -> str:
    """Find weather of given state"""
    print(f"#######{state}###########")
    return f"It's Sunny day in {state}"


#The transport="stdio" argument tells the server to:

#Use standard input/output (stdin and stdout) to receive and respond to tool function calls.

if __name__=="__main__":
    mcp.run(transport="stdio")