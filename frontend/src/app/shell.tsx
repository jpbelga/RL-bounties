"use client";
import { AppShell } from "@mantine/core";
import { FC } from "react";
import { Header } from "./header";
import { Footer } from "./footer";
import { useAccount } from "wagmi";

export const Shell: FC<{ children: React.ReactNode }> = ({ children }) => {
    const { address, connector, isConnected } = useAccount();

    const offset = !!address && !!connector && !!isConnected ? "10rem" : "7rem";

    return (
        <AppShell header={{ height: "auto" }}>
            <AppShell.Header>
                <Header />
            </AppShell.Header>
            <AppShell.Main mt={{ base: offset, sm: "7rem" }} pb="8rem">
                {children}
            </AppShell.Main>
            <AppShell.Footer>
                <Footer />
            </AppShell.Footer>
        </AppShell>
    );
};
