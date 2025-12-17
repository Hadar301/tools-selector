from time import time

from intialize_agent import initialize_agent, initialize_filtered_agent


def main():
    agent_executor = initialize_agent()
    filtered_agent_executor = initialize_filtered_agent()

    query1 = "What is the weather in London?"
    query2 = "what is the prices of the stock INTC?"
    query3 = "How much is 6 times 900 ?"
    message_to_agent = query1 + "\n" + query3

    start_time = time()
    response = filtered_agent_executor.invoke({"messages": [("user", message_to_agent)]})
    print(f"\n--- Final Response ---\n took about {time() - start_time:.3f} seconds")
    print(response["messages"][-1].content)

    start_time = time()
    response = agent_executor.invoke({"messages": [("user", message_to_agent)]})
    print(f"\n--- Final Response ---\n took about {time() - start_time:.3f} seconds")
    print(response["messages"][-1].content)

    start_time = time()
    response = filtered_agent_executor.invoke({"messages": [("user", message_to_agent)]})
    print(f"\n--- Final Response ---\n took about {time()-start_time :.3f} seconds")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
